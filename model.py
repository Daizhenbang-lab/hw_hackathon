import torch.nn as nn
import torch
from torch import Tensor
from typing import Type


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module = None,
    ) -> None:
        super(BasicBlock, self).__init__()
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        # print(out.shape)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # print(out.shape)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        # print(out.shape, identity.shape)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        img_channels: int = 1,
        out_channels: int = 2,
        block: Type[BasicBlock] = BasicBlock,
    ) -> None:
        super(ResNet, self).__init__()

        layers = [2, 2, 2, 2]
        self.expansion = 1

        scaling_factor = 2

        self.in_channels = 64 // scaling_factor

        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 5), stride=2, padding=1)

        self.layer1 = self._make_layer(block, self.in_channels, layers[0])
        self.layer2 = self._make_layer(block, 128 // scaling_factor, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256 // scaling_factor, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 // scaling_factor, out_channels)

    def _make_layer(
        self, block: Type[BasicBlock], out_channels: int, blocks: int, stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = []
        layers.append(
            block(self.in_channels, out_channels, stride, self.expansion, downsample)
        )
        self.in_channels = out_channels * self.expansion

        for _ in range(blocks):
            layers.append(
                block(self.in_channels, out_channels, expansion=self.expansion)
            )
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            x = self.transform(x, augment=self.training)

        x = self.conv1(x)
        # print(x.shape)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def train_loss(self, output, target):
        loss = nn.functional.mse_loss(output, target)
        return loss

    def test_loss(self, output, target):
        x = output
        return nn.functional.pairwise_distance(x, target).mean()

    def transform(self, data, augment):
        # data augmentation by shifting the data in the time domain to keep robustness against time advance
        if augment:
            leave_out = 10
            select_start_pos = torch.randint(0, leave_out, (data.shape[0],))
            place_start_pos = torch.randint(0, leave_out, (data.shape[0],))
            num_units = 408 - leave_out
            new_data = torch.zeros_like(data)
            for i in range(data.shape[0]):
                new_data[i, :, :, place_start_pos[i]:place_start_pos[i]+num_units] = data[i, :, :, select_start_pos[i]:select_start_pos[i]+num_units]
            data = new_data

        # fft
        data = data.view(data.shape[0], 4, -1, 408)
        data = data[:, :2, :, :] + 1j * data[:, 2:, :, :]
        data = torch.fft.fftn(data, dim=[-1])

        data_r, data_i = data.real, data.imag
        data = torch.concatenate([data_r, data_i], axis=1) # shape (n, 4, 64, 408)
        data = data.view(data.shape[0], 1, -1, data.shape[-1]) # shape (n, 1, 4*64, 408)
        return data


if __name__ == "__main__":
    x = torch.rand([3, 1, 128, 408])
    model = ResNet()

    # show number of parameters
    print(sum(p.numel() for p in model.parameters()))

    output = model(x)
