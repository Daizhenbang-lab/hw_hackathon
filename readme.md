Please cache the data first to avoid the I/O bottleneck.
1. Make sure you place the zip files, cfg files and input position files in folder `./data` first. Then by running `python3 cache_H_data.py` we cache the H data in both time and frequency domain as npy files.

2. Then simply run `python3 train_and_infer.py --dataset_idx 2 --data_idx 1` for both training and inferring. Feel free to increase the number of epochs: Due to the heavy regularisation and data augmentation, it is unlikely to encounter "bad overfitting" in general.

Note: Make sure you have pytorch with cuda support installed properly to run the code.