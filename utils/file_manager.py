import pickle
from models import pixelDDN
import torch

def pickle_save(data, path):
    ''' saves saved_data to path in .pkl format
    Params:
        saved_data: saved_data to save
        path (str): path for saving
    '''
    # write pickle representation of saved_data to file
    # write in highest protocol version --> best speed up
    with open(path, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

def pickle_load(path):
    ''' load a .pkl file from path
    Params:
        path (str): path of file
    Return:
        saved_data (obj): reconstituted object
        '''
    data = None
    # open in binary format for reading
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def load_model(modelname, data_config, PATH):
    """ loads the specified model
    Params:
        modelname (str): contains name of the model
        path (str): path to the models parameters
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

