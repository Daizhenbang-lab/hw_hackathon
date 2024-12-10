import numpy as np
import zipfile
import os


def cache_H_data(dataset_idx, data_idx):
    # source file names
    cfg_path = f'data/Dataset{dataset_idx}CfgData{data_idx}.txt'
    if dataset_idx == 0:
        input_data_path = f'data/Round{dataset_idx}InputData{data_idx}.txt'
    elif dataset_idx == 1:
        input_data_path = f'data/Dataset{dataset_idx}InputData{data_idx}.txt'
    elif dataset_idx == 2:
        input_data_path = f'data/Round{dataset_idx+1}InputData{data_idx}.txt'
    else:
        raise ValueError(f'Invalid dataset index {dataset_idx}')

    # target file names
    Hf_file = f'data/Hf{dataset_idx}_{data_idx}.npy'
    Ht_file = f'data/Ht{dataset_idx}_{data_idx}.npy'

    # read cfg file
    info = np.loadtxt(cfg_path, skiprows=1)
    info = info.astype(int)
    _, _, port_num, ant_num, sc_num = tuple(info.tolist())

    Hf = np.loadtxt(input_data_path)
    Hf = np.reshape(Hf, (-1, 2, sc_num, ant_num, port_num)) # n*2*408*64*2

    # recover the complex data
    Hf_real = Hf[:, 0, :, :, :].transpose(0, 3, 2, 1)
    Hf_imag = Hf[:, 1, :, :, :].transpose(0, 3, 2, 1)
    Hf = Hf_real + 1j * Hf_imag

    # ifft along the last dimension
    Ht = np.fft.ifft(Hf, axis=-1)

    # save the data in float16
    Hf = np.stack((Hf.real, Hf.imag), axis=-1).astype(np.float16)
    Ht = np.stack((Ht.real, Ht.imag), axis=-1).astype(np.float16)

    np.save(Hf_file, Hf)
    np.save(Ht_file, Ht)

    # remove the unzipped file
    os.remove(input_data_path)
    print(f'Removed {input_data_path}')


if __name__ == '__main__':
    dataset_idx = 2
    data_indices = [1, 2, 3]

    for data_idx in data_indices:
        zip_file = f'data/Dataset{dataset_idx}InputData{data_idx}.zip'

        # unzip the file
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall('data/')

        print(f'Unzipped {zip_file}')

        # cache the data
        cache_H_data(dataset_idx, data_idx)

        print(f'Cached {zip_file}\n\n')
