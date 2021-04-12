import pickle
from models import pixelDDN
import torch

# class for saving and loading data sets as well as the model parameters of trained models

def pickle_save(data, path):
    """ saves data to path in .pkl format
    Params:
        saved_data (dict): data to save
        path (string): path for saving
    """
    # write pickle representation of saved_data to file
    # write in highest protocol version --> best speed up
    with open(path, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

def pickle_load(path):
    """ loads a .pkl file from path
    Params:
        path (str): path of file
    Return:
        data (dict): reconstituted object
    """
    # open in binary format for reading
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def load_model(modelname, data_config, PATH):
    """ loads the specified model
    Params:
        modelname (string): contains name of the model
        path (string): path to the models parameters
    Returns:
        model (pixelDDN object): model
    """
    names = modelname.split("_")
    ddn_model = names[0]
    ae_model = names[1]

    path = f"{PATH}/ae_config"
    ae_config = pickle_load(path)
    path = f"{PATH}/ddn_config"
    ddn_config = pickle_load(path)

    # init PixelDDN network
    if ddn_model == "MLP":
        model = pixelDDN.PixelMLP(ae_model, ddn_model, ae_config, ddn_config, data_config, data_config['latent_dim'])
    if ddn_model == "LNN":
        model = pixelDDN.PixelLNN(ae_model, ddn_model, ae_config, ddn_config, data_config, data_config['latent_dim'])
    if ddn_model == "HNN":
        model = pixelDDN.PixelHNN(ae_model, ddn_model, ae_config, ddn_config, data_config, data_config['latent_dim'])
    if ddn_model == "VIN":
        raise NotImplementedError

    # load saved model parameters
    path = f"{PATH}/{modelname}"
    model.load_state_dict(torch.load(path))

    return model