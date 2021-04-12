import torch
import yaml
import numpy as np

# class containing different helper functions needed in the process

def choose_activation(name):
    """ returns torch activation function corresponding to the desired one
    Params:
        name (str): name of activation
    Returns:
        act: torch activation
    """
    if name == 'tanh':
        act = torch.tanh
    elif name == 'relu':
        act = torch.relu
    elif name == 'relu^2':
    	act = lambda x: torch.relu(torch.pow(x, 2))
    elif name == 'sigmoid':
        act = torch.sigmoid
    elif name == 'softplus':
        act = torch.nn.functional.softplus
    elif name == 'selu':
        act = torch.nn.functional.selu
    elif name == 'elu':
        act = torch.nn.functional.elu
    elif name == 'gelu':
        act = torch.nn.functional.gelu
    elif name == 'swish':
        act = lambda x: x * torch.sigmoid(x)
    else:
        raise ValueError("Did not find activation function")
    return act

def read_config(config_file):
    """ reads a config file and returns as dictionary
    Params:
        config_file (str): path to config file
    Returns:
        config (dict): config file as dictionary
    """
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def convert_data(data):
    """ constructs training set and test set from data
    Params:
        data (dict): dictionary containing all data
    Returns:
        trainset, testset: corresponding training set and test set
    """
    trainset = data['train_data']
    testset = data['test_data']

    # stack all sequences vertically
    trainset = np.vstack(trainset)
    testset = np.vstack(testset)

    # convert to channels first
    trainset = torch.from_numpy(trainset)
    trainset = trainset.permute(0, 1, 4, 2, 3)
    testset = torch.from_numpy(testset)
    testset = testset.permute(0, 1, 4, 2, 3)

    return trainset, testset