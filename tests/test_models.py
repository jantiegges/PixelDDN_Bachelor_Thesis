########
# class that has been used for testing the models
########


import numpy as np
import torch
from os import path

from models import autoencoder
from models import dynamic_net
from models import pixelDDN
from utils import helper


class Test_Autoencoder:
    """ class that contains functions for testing the autoencoder """

    def __init__(self, ae_params, batch_size, channels, seq_len, im_size, latent_dim, model):
        """
        ae_params (dict): contains network parameters for autoencoder
        batch_size (int): size of batch
        channels (int): number of channels of the input images (RGB: 3, BW: 1)
        seq_len (int): number of images in one sequence
        im_size (int): Height/Width of input image in pixel
        latent_dim (int): dimension of latent space
        model (str): name of the autoencoder model to test
        """
        self.ae_params = ae_params
        self.en_params = ae_params['encoder']
        self.de_params = ae_params['decoder']
        self.batch_size = batch_size
        self.channels = channels
        self.seq_len = seq_len
        self.im_size = im_size
        self.latent_dim = latent_dim
        self.model = model

        if model == "LAE":
            self.ae = autoencoder.LAE(self.en_params, self.de_params, channels, seq_len, im_size, latent_dim)
        if model == "CAE":
            raise NotImplementedError
        if model == "VAE":
            raise NotImplementedError
        if model == "CVAE":
            self.ae = autoencoder.CVAE(self.en_params, self.de_params, channels, seq_len, im_size, latent_dim)

    def test_encoder_out_shape(self):
        ''' tests if the encoder is working and delivering the right output shape '''

        # set the expected size of the encoder output
        expected_out_size = torch.Size([batch_size, latent_dim])
        print(f"expected out size: {expected_out_size}")
        # create random images
        rand_images = np.random.randint(0, 255, size=(batch_size, seq_len, channels, im_size, im_size))

        if self.ae_params["variational"]:
            # reshape images to fit input
            rand_images = rand_images.reshape((batch_size, seq_len * channels, im_size, im_size))
            inputs = torch.tensor(rand_images).float()
            print("input size: {}".format(inputs.size()))

            # call encoder
            mean, log_var = self.ae.encode(inputs)
            z = self.ae.reparameterize(mean, log_var)
        else:
            # reshape images to fit input
            rand_images = rand_images.reshape((batch_size, seq_len * channels * im_size * im_size))
            inputs = torch.tensor(rand_images).float()
            print("input size: {}".format(inputs.size()))

            # call encoder
            z = self.ae.encode(inputs)

        # print output size and check if it is the same as expected
        print(f"size latent space: {z.size()}")
        print(f"random latent representation: {z[0]}")
        assert z.size() == expected_out_size

    def test_decoder_out_shape(self):
        ''' tests if the encoder is working and delivering the right output shape '''

        # set the expected size of the encoder output
        expected_out_size = torch.Size([batch_size, channels, im_size, im_size])
        # create random latent position as input
        rand_latent_p = np.random.randint(0, np.pi, size=(batch_size, latent_dim / 2))
        inputs = torch.tensor(rand_latent_p).float()
        print(f"expected out size: {expected_out_size}")
        print(f"latent size of p: {inputs.size()}")

        # call encoder
        x_hat = self.ae.decode(inputs)

        # print output size and check if it is the same as expected
        print(f"size output image: {x_hat.size()}")
        print(f"random output image: {x_hat[0]}")
        assert x_hat.size() == expected_out_size


class Test_DDN():
    """ class that contains functions for testing the dynamic network """

    def __init__(self, dnn_params, batch_size, latent_dim, model):
        """
        ddn_params (dict): contains network parameters for the dynamic network
        batch_size (int): size of batch
        latent_dim (int): dimension of latent space
        model (str): name of the dynamic network model to test
        """
        self.ddn_params = dnn_params
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.model = model

        if model == "MLP":
            raise NotImplementedError
        if model == "LNN":
            raise NotImplementedError
        if model == "HNN":
            self.dnn = dynamic_net.HNN(dnn_params, latent_dim)
        if model == "VIN":
            raise NotImplementedError

    def test_ddn_out_shape(self):
        """ tests if the dynamic network is working and delivering the right output shape """

        # create random latent position as input
        rand_latent = np.random.randint(0, np.pi, size=(batch_size, latent_dim))
        inputs = torch.tensor(rand_latent).float()
        print(f"latent size: {inputs.size()}")

        # set the expected size of the dynamics network output
        if self.model == "MLP":
            expected_out_size = torch.Size([batch_size, latent_dim])
        if self.model == "LNN":
            # TODO: what is size of q_tt
            expected_out_size = torch.Size([batch_size, latent_dim / 2])
        if self.model == "HNN":
            expected_out_size = torch.Size([batch_size, 1])
        if self.model == "VIN":
            # TODO: what is size of VIN Output
            raise NotImplementedError
        print(f"expected out size: {expected_out_size}")

        x = self.dnn(inputs)

        # print output size and check if it is the same as expected
        print(f"size of dynamic network output: {x.size()}")
        print(f"random ddn output: {x[0]}")
        assert x.size() == expected_out_size


class Test_PixelDDN():
    """ class that contains functions for testing the whole Pixel Dynamics network """

    def __init__(self, ae_model, ddn_model, ae_params, ddn_params, data_params, batch_size, latent_dim):
        self.ae_model = ae_model
        self.ddn_model = ddn_model
        self.ae_params = ae_params
        self.ddn_params = ddn_config
        self.data_params = data_params
        self.batch_size = batch_size
        self.channels = data_params['channels']
        self.seq_len = data_params['seq_len']
        self.im_size = data_params['im_size']
        self.latent_dim = latent_dim

        if ddn_model == "MLP":
            self.model = pixelDDN.PixelMLP(ae_model, ddn_model, ae_params, ddn_params, data_params, latent_dim)
        if ddn_model == "LNN":
            self.model = pixelDDN.PixelLNN(ae_model, ddn_model, ae_params, ddn_params, data_params, latent_dim)
        if ddn_model == "HNN":
            self.model = pixelDDN.PixelHNN(ae_model, ddn_model, ae_params, ddn_params, data_params, latent_dim)
        if ddn_model == "VIN":
            raise NotImplementedError


    def test_pixelddn_out_shape(self):
        """ tests if the network is working and delivering the right output shape """

        # set the expected size of the encoder output
        expected_out_size = torch.Size([batch_size, channels, im_size, im_size])
        print(f"expected out size: {expected_out_size}")
        # create random images
        rand_images = np.random.randint(0, 255, size=(batch_size, seq_len, channels, im_size, im_size))

        inputs = torch.tensor(rand_images).float()
        print("input size: {}".format(inputs.size()))

        if self.ae_params['variational']:
            pred = self.model(inputs)
        else:
            pred = self.model(inputs, variational=False)

        # print output size and check if it is the same as expected
        print(f"size reconstruction: {pred.reconstruction.size()}")
        print(f"size of latent space: {pred.z.size()}")
        assert pred.reconstruction[0].size() == expected_out_size


if __name__ == '__main__':
    """ Test functions:
    Test Autoencoder:
        test_encoder_out_shape(): tests if the encoder is working and delivering the right output shape
        test_decoder_out_shape(): tests if the decoder is working and delivering the right output shape

    Test DNN:
        test_dnn_out_shape():

    Test PixelDNN:
        test_pixeldnn_out_shape():
    """

    basepath = path.dirname(__file__)

    # read configuration files
    filepath = path.abspath(path.join(basepath, '..', 'config/model_config.yaml'))
    model_config = helper.read_config(filepath)
    filepath = path.abspath(path.join(basepath, '..', 'config/train_config.yaml'))
    train_config = helper.read_config(filepath)
    filepath = path.abspath(path.join(basepath, '..', 'config/data_config.yaml'))
    data_config = helper.read_config(filepath)

    # Set network models to test
    ae_model = 'LAE'
    ddn_model = 'MLP'
    latent_dim = 2

    # initialise autoencoder and dynamics network configurations
    ae_config = model_config['autoencoder'][ae_model]
    ddn_config = model_config['dynamics'][ddn_model]

    batch_size = 20
    channels = data_config['channels']
    seq_len = data_config['seq_len']
    im_size = data_config['im_size']

    # init network test objects
    ae = Test_Autoencoder(ae_config, batch_size, channels, seq_len, im_size, latent_dim, model=ae_model)
    # ddn = Test_DDN(ddn_config, batch_size, latent_dim, model=ddn_model)
    pixelddn = Test_PixelDDN(ae_model, ddn_model, ae_config, ddn_config, data_config, batch_size, latent_dim)

    # print("Test encoder output shape\n")
    #ae.test_encoder_out_shape()

    # print("\nTest DDN output shape\n")
    # ddn.test_ddn_out_shape()

    pixelddn.test_pixelddn_out_shape()
