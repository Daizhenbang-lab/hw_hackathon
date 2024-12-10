from datasets import load_labelled_data, load_all_data
from model import ResNet
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import os
import numpy as np
import argparse


# set gpu device
def main(args):
    torch.cuda.set_device(args.gpu)

    # load the dataset
    dataset_idx = args.dataset_idx
    data_idx = args.data_idx

    os.makedirs('checkpoints', exist_ok=True)

    # Load the time domain data
    train_set = load_labelled_data(
        f'data/Ht{dataset_idx}_{data_idx}.npy',
        f'data/Dataset{dataset_idx}InputPos{data_idx}.txt')
    test_set = load_all_data(f'data/Ht{dataset_idx}_{data_idx}.npy')

    # relatively large batch size for reducing the variance of the gradient estimates
    # not using the full data as a batch to introduce stochasticity for the training to improve generalization
    train_loader = DataLoader(
        train_set, batch_size=256, shuffle=True,
        num_workers=8, pin_memory=True, prefetch_factor=2)

    test_loader = DataLoader(
        test_set, batch_size=512, shuffle=False)

    # initialize the model
    model = ResNet().cuda()

    ########################### TRAINING ###########################

    num_epochs = args.num_epochs

    # Heavy l2 regularization to improve generalization
    optimizer = optim.AdamW(
        model.parameters(), lr=3e-3, weight_decay=5e-2)

    # for more stable training
    scheduler = CosineAnnealingLR(
        optimizer, T_max=len(train_loader)*num_epochs, eta_min=1e-5)

    for epoch in range(num_epochs):
        model.train()
        error = 0
        for input, target in train_loader:
            input, target = input.cuda(), target.cuda()
            optimizer.zero_grad()
            loss = model.train_loss(model(input), target)
            loss.backward()
            error += loss.item()
            optimizer.step()
            scheduler.step()

        error /= len(train_loader)
        print(f'Epoch {epoch}, Train error: {error}', flush=True)

        torch.save(model.state_dict(), f'checkpoints/{dataset_idx}_{data_idx}.pth')

    ########################### INFERENCE ###########################

    # test the model
    with torch.no_grad():
        model.eval()
        ys = []
        for input in test_loader:
            input = input.cuda()
            ys.append(model(input).cpu().numpy())
        ys = np.concatenate(ys)

    # replace the prediction with the ground truth
    given_data = np.loadtxt(f'data/Dataset{dataset_idx}InputPos{data_idx}.txt')
    indices = given_data[:, 0].astype(int) - 1
    given_data = given_data[:, 1:]
    ys[indices] = given_data

    # save the result in 4-digit precision
    os.makedirs('results', exist_ok=True)
    np.savetxt(f'results/Dataset{dataset_idx}Output{data_idx}.txt', ys, fmt='%.4f')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset_idx', type=int, default=2)
    parser.add_argument('--data_idx', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=500)

    args = parser.parse_args()
    main(args)
