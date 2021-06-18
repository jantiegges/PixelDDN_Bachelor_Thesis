########
# class that has been used for testing the data generation
########


import numpy as np
from os import path
from IPython import display
from data import get_dataset
from utils.figures import make_gif
from utils.helper import read_config
import torch
import tqdm


if __name__ == '__main__':
    i1 = np.array([[2, 3, 3],
                   [2, 1, 1],
                   [4, 4, 1]])

    i2 = np.array([[3, 3, 1],
                   [4, 3, 4],
                   [3, 1, 3]])

    vertical = np.concatenate([i1[:-1], i1[1:]], axis=-1)

    basepath = path.dirname(__file__)
    filepath = path.abspath(path.join(basepath, '..', 'config/data_config.yaml'))
    data_dir = path.join(basepath, "../saved_data")
    data_config = read_config(filepath)
    system_name = "pendulum"
    filename = "pendulum_5_40_64_4_bw"

    data = get_dataset(data_config, system_name, data_dir, filename, seed=32)

    im_size = data['settings']['im_size']
    episodes = data['settings']['episodes']
    timesteps = data['settings']['timesteps']
    num_frames = data['train_meta_data']['timesteps']

    # takes frames of first episode
    frames = data['train_data'][1,:,0,:,:,0]

    # creates and saves gif
    path = "./figures"
    gifname = make_gif(frames, path, duration=1e-1, pixels=[120, 120])

    trainset = data['train_data']
    # stack all sequences vertically
    trainset = np.vstack(trainset)

    # convert to channels first
    trainset = torch.from_numpy(trainset)
    trainset = trainset.permute(0, 1, 4, 2, 3)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)


    pbar = tqdm.tqdm(trainloader)

    for batch_idx, rollout_batch in enumerate(pbar):

        seq_len = 4
        target = rollout_batch[:,-1,...]
        rollout = rollout_batch[:,:seq_len,...]
        dummy = 1

    seq_len = data['settings']