import torch
import torch.nn as tnn
from utils import helper

"""
Models:
    LAE: Simple Linear Fully Connected Autoencoder
    CAE: Convolutional Autoencoder
    VAE: Variational Autoencoder
    CVAE: Convolutional Variational Autoencoder
"""

class AE(torch.nn.Module):
    """ Basic Autoencoder model: All other models inherit from this one"""

    def __init__(self, en_params, de_params, channels, seq_len, im_size, latent_dim, activation="relu"):
        """
        Params:
                en_params (dict): contains model parameters for the encoder
                de_params (dict): contains model parameters for the decoder
                channels (int): number of channels of the input images (RGB: 3, BW: 1)
                seq_len (int): number of images in one sequence
                im_size (int): Height/Width of input image in pixel
                latent_dim (int): dimension of latent space
        """
        super(AE, self).__init__()

        self.en_params = en_params
        self.de_params = de_params

        # set other params
        self.latent_dim = latent_dim

        # only for normal autoencoder, Convolutional Autoencoder have Leaky Relu as activation
        self.activation = helper.choose_activation(activation)

        self._build_network(im_size, channels, seq_len)

    def _build_network(self, **kwargs):
        """ sets the parameters for the network models and calls _build_encoder and _build_decoder """
        raise NotImplementedError()

    def _build_encoder(self, **kwargs):
        """ builds the network components of the encoder """
        raise NotImplementedError()

    def _build_decoder(self, **kwargs):
        """ builds the network components of the decoder """
        raise NotImplementedError()

    def encode(self, **kwargs):
        """ encodes the sequence of input images to a latent representation """
        raise NotImplementedError()

    def decode(self, **kwargs):
        """ decodes a latent representation to an image """
        raise NotImplementedError()

    def forward(self, **kwargs):
        """ Sets forward pass """
        raise NotImplementedError()


class LAE(AE):
    """ Basic fully-connected linear autoencoder"""

    def _build_network(self, im_size, channels, seq_len):
        """
        Params:
                im_size (int): Height/Width of input image in pixel
                channels (int): number of channels of the input images (RGB: 3, BW: 1)
                seq_len (int): number of images in one sequence
        """

        en_in_features = (im_size ** 2) * seq_len * channels
        en_hidden_layers = self.en_params["hidden_layers"]
        en_hid_features = self.en_params["out_features"]
        en_out_features = self.latent_dim

        de_in_features = int(self.latent_dim/2)
        de_hidden_layers = self.de_params["hidden_layers"]
        de_hid_features = self.de_params["out_features"]
        de_out_features = (im_size ** 2) * channels

        self._build_encoder(en_in_features, en_hidden_layers, en_hid_features, en_out_features)
        self._build_decoder(de_in_features, de_hidden_layers, de_hid_features, de_out_features)

    def _build_encoder(self, in_features, hidden_layers, hid_features, out_features):
        """ builds a linear encoder network
        Params:
            in_features (int): size of input sequence
            hidden_layers (int): number of hidden layers
            hid_features (list[int]): list with number of hidden neurons for each layer
            out_features (int): latent dimension
        """
        # input layer
        self.encoder = tnn.ModuleList([tnn.Linear(in_features, hid_features[0])])
        # hidden layer
        self.encoder.extend([tnn.Linear(hid_features[h], hid_features[h + 1]) for h in range(hidden_layers)])
        # output layer
        self.encoder.append(tnn.Linear(hid_features[-1], out_features))

    def _build_decoder(self, in_features, hidden_layers, hid_features, out_features):
        """ builds a linear decoder network
        Params:
            in_features (int): size of latent position
            hidden_layers (int): number of hidden layers
            hid_features (list[int]): list with number of hidden neurons for each layer
            out_features (int): size of one image
        """
        # input layer
        self.decoder = tnn.ModuleList([tnn.Linear(in_features, hid_features[0])])
        # hidden layer
        self.decoder.extend([tnn.Linear(hid_features[h], hid_features[h + 1]) for h in range(hidden_layers)])
        # output layer
        self.decoder.append(tnn.Linear(hid_features[-1], out_features))

    def encode(self, x):
        """
        Params:
            x (Tensor): contains flattened sequence of images
        Returns:
            x (Tensor): contains latent encoding of input
        """
        for layer in self.encoder:
            x = self.activation(layer(x))
        return x

    def decode(self, z):
        """
        Params:
            z (Tensor) [N, x]: contains latent encoding for last image of the sequence
        Returns:
            z (Tensor) [N, latent_dim]: contains flattened reconstructed image
        """
        for layer in self.decoder:
            z = self.activation(layer(z))

        return z

    def forward(self, x):
        """ encodes image sequence into latent space and reconstructs last image again
        Params:
            x (Tensor): contains flattened sequence of images
        Returns:
            x_hat (Tensor): contains flattened reconstructed image (last one of sequence)
        """
        z = self.encode(x)
        q = z[:, :int(self.latent_dim / 2)]
        x_hat = self.decode(q)
        return x_hat


class CAE(AE):
    """ Convolutional Autoencoder """

    def _build_network(self, im_size, channels, seq_len):
        """
        Params:
            im_size (int): Height/Width of input image in pixel
            channels (int): number of channels of the input images (RGB: 3, BW: 1)
            seq_len (int): number of images in one sequence
        """
        self.channels = channels
        # compute number of input channels
        self.input_channels = channels * seq_len
        self.im_size = im_size

        # set params for encoder and decoder
        en_hidden_layers = self.en_params["hidden_layers"]
        en_filters = self.en_params["filters"]
        en_kernel_sizes = self.en_params["kernel_sizes"]
        en_strides = self.en_params["strides"]
        en_paddings = [int(k / 2) for k in en_kernel_sizes]

        de_hidden_layers = self.de_params["hidden_layers"]
        self.de_filters = self.de_params["filters"]
        de_kernel_sizes = self.de_params["kernel_sizes"]
        de_strides = self.de_params["strides"]
        de_paddings = [int(k / 2) for k in de_kernel_sizes]

        # call build function to construct encoder and decoder
        self._build_encoder(en_filters, en_kernel_sizes, en_strides, en_paddings, en_hidden_layers)
        self._build_decoder(self.de_filters, de_kernel_sizes, de_strides, de_paddings, de_hidden_layers)

    def _build_encoder(self, filters, kernel_sizes, strides, paddings, hidden_layers):
        """ builds a convolutional encoder network
        Params:
            filters (list[int]): size of input/output space
            kernel_sizes (list[int]): specify the height/width of the 2D convolution window
            strides (list[int]): specify the stride of the convolution
            paddings (list[int]): specify how the filter should behave when it hits the edge of the matrix
            hidden_layers (int): number of hidden layers
        """

        # create module list
        modules = []
        in_channels = self.input_channels

        # append first and hidden layers to the list
        for h in range(hidden_layers + 1):
            modules.append(
                tnn.Sequential(
                    tnn.Conv2d(in_channels,
                               out_channels=filters[h],
                               kernel_size=kernel_sizes[h],
                               stride=strides[h],
                               padding=paddings[h]),
                    tnn.BatchNorm2d(filters[h]),
                    tnn.LeakyReLU())
            )
            in_channels = filters[h]

        self.encoder = tnn.Sequential(*modules)

        # specify output layers
        self.out = tnn.Linear(filters[-1]*4, self.latent_dim)

    def _build_decoder(self, filters, kernel_sizes, strides, paddings, hidden_layers):
        """ builds a convolutional encoder network
        Params:
            filters (list[int]): size of input/output space
            kernel_sizes (list([int]): specify the height/width of the 2D convolution window
            strides (list[int]): specify the stride of the convolution
            paddings (list[int]): specify how the filter should behave when it hits the edge of the matrix
            hidden_layers (int): number of hidden layers
        """

        modules = []

        # compute input size to network depending on latent space
        self.decoder_in = tnn.Linear(int(self.latent_dim / 2), filters[0] * 4)

        # append hidden layers to the module list
        for h in range(hidden_layers):
            modules.append(
                tnn.Sequential(
                    tnn.ConvTranspose2d(filters[h],
                                        filters[h+1],
                                        kernel_size=kernel_sizes[h],
                                        stride=strides[h],
                                        padding=paddings[h],
                                        output_padding=paddings[h]),
                    tnn.BatchNorm2d(filters[h+1]),
                    tnn.LeakyReLU())
            )

        self.decoder = tnn.Sequential(*modules)

        # construct output layer (no activation)
        self.decoder_out = tnn.Sequential(
                            tnn.ConvTranspose2d(filters[-1],
                                                filters[-1],
                                                kernel_size=kernel_sizes[-1],
                                                stride=strides[-1],
                                                padding=paddings[-1],
                                                output_padding=paddings[-1]),
                            tnn.BatchNorm2d(filters[-1]),
                            tnn.LeakyReLU(),
                            tnn.Conv2d(filters[-1],
                                       out_channels=self.channels,
                                       kernel_size=kernel_sizes[-1],
                                       padding=paddings[-1]),
                            tnn.Tanh())

    def encode(self, x):
        """ Encodes a sequence of images to a latent state representation
        Params:
            x (Tensor) [batch_size, seq_len*in_channels, height, width]: contains sequence of frames
        Returns:
            z (Tensor) [N, latent_dim]: latent state representation
        """
        z = self.encoder(x)
        # flatten the output of convolutional network
        z = torch.flatten(z, start_dim=1)
        z = self.out(z)

        return z

    def decode(self, q):
        """ decodes an image from a latent position variable
        Params:
            q (Tensor)[N, latent_dim]: latent position variable
        Returns:
            x_hat (Tensor)[N, C, H, W]: single reconstructed output image
        """
        x_hat = self.decoder_in(q)
        # reshape to fit to first hidden layer
        x_hat = x_hat.view(-1, self.de_filters[0], 2, 2)
        x_hat = self.decoder(x_hat)
        x_hat = self.decoder_out(x_hat)
        return x_hat

    def forward(self, x):
        """
        Params:
            x (Tensor) [N, S*C, H, W]: contains sequence of frames
        Returns:
            x_hat (Tensor)[N, C, H, W]: single reconstructed output image
        """
        z = self.encode(x)
        q = z[:, :int(self.latent_dim / 2)]
        x_hat = self.decode(q)
        return x_hat


class VAE(AE):
    """ Variational Autoencoder """

    def _build_network(self, im_size, channels, seq_len):
        """
        Params:
                im_size (int): Height/Width of input image in pixel
                channels (int): number of channels of the input images (RGB: 3, BW: 1)
                seq_len (int): number of images in one sequence
        """

        en_in_features = (im_size ** 2) * seq_len * channels
        en_hidden_layers = self.en_params["hidden_layers"]
        en_hid_features = self.en_params["out_features"]
        en_out_features = self.latent_dim

        de_in_features = int(self.latent_dim/2)
        de_hidden_layers = self.de_params["hidden_layers"]
        de_hid_features = self.de_params["out_features"]
        de_out_features = (im_size ** 2) * channels

        self._build_encoder(en_in_features, en_hidden_layers, en_hid_features, en_out_features)
        self._build_decoder(de_in_features, de_hidden_layers, de_hid_features, de_out_features)

    def _build_encoder(self, in_features, hidden_layers, hid_features, out_features):
        """ builds a linear encoder network that maps input to mu with two output layers for mean and variance
        Params:
            in_features (int): size of input sequence
            hidden_layers (int): number of hidden layers
            hid_features (list[int]): list with number of hidden neurons for each layer
            out_features (int): latent dimension
        """

        # input layer
        self.encoder = tnn.ModuleList([tnn.Linear(in_features, hid_features[0])])
        # hidden layer
        self.encoder.extend([tnn.Linear(hid_features[h], hid_features[h+1]) for h in range(hidden_layers)])
        # output layer
        self.fc_mean = tnn.Linear(hid_features[-1], out_features)
        self.fc_logvar = tnn.Linear(hid_features[-1], out_features)

    def _build_decoder(self, in_features, hidden_layers, hid_features, out_features):
        """ builds a linear decoder network
        Params:
            in_features (int): size of latent position
            hidden_layers (int): number of hidden layers
            hid_features (list[int]): list with number of hidden neurons for each layer
            out_features (int): size of one image
        """

        # input layer
        self.decoder = tnn.ModuleList([tnn.Linear(in_features, hid_features[0])])
        # hidden layer
        self.decoder.extend([tnn.Linear(hid_features[h], hid_features[h+1]) for h in range(hidden_layers)])

        # output layer
        self.output = tnn.Linear(hid_features[-1], out_features)

    def encode(self, x):
        """
        Params:
            x (Tensor) [N, x]: contains flattened sequence of images
        Returns:
            mu (Tensor) [N, latent_dim]: contains encoded mean of input
            logvar (Tensor) [N, latent_dim]: contains variance of input
        """
        for layer in self.encoder:
            x = self.activation(layer(x))

        # Split the result into mu and var of the latent Gaussian distribution
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)

        return mean, logvar

    def decode(self, z):
        """
        Params:
            z (Tensor) [N, latent_dim]: contains latent encoding for last image of the sequence
        Returns:
            z (Tensor) [N, x]: contains flattened reconstructed image
        """
        for layer in self.decoder:
            z = self.activation(layer(z))

        return z

    def reparameterize(self, mean, logvar):
        """ gets sample from N(mu, var) with reparameterisation trick
        Params:
            mu (Tensor) [N, latent_dim]: contains encoded mean of input
            logvar (Tensor) [N, latent_dim]: contains variance of input
        Returns:
            z (Tensor) [N, latent_dim]: contains latent encoding for one image
        """
        stddev = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(stddev)
        z = epsilon * stddev + mean
        return z

    def forward(self, x):
        """ encodes input sequence into mean and variance for last image and reconstructs it again
        Params:
            x (Tensor) [N, x]: contains flattened sequence of images
        Returns:
            x_hat (Tensor) [N, y]: contains flattened reconstructed image (last one of sequence)
        """
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        q = z[:, :int(self.latent_dim / 2)]
        x_hat = self.decode(q)
        return x_hat


"""
Code adapted from:
    - @article{toth2019hamiltonian,
      title={Hamiltonian generative networks},
      author={Toth, Peter and Rezende, Danilo Jimenez and Jaegle, Andrew and Racani{\`e}re, S{\'e}bastien and Botev, Aleksandar and Higgins, Irina},
      journal={arXiv preprint arXiv:1909.13789},
      year={2019}
    }
    - https://www.tensorflow.org/tutorials/generative/cvae
    - https://github.com/AntixK/PyTorch-VAE/blob/master/models/beta_vae.py
"""

class CVAE(AE):
    """ Convolutional Variational Autoencoder """

    def _build_network(self, im_size, channels, seq_len):
        """
        Params:
                im_size (int): Height/Width of input image in pixel
                channels (int): number of channels of the input images (RGB: 3, BW: 1)
                seq_len (int): number of images in one sequence
        """
        self.channels = channels
        # compute number of input channels
        self.input_channels = channels * seq_len
        self.im_size = im_size

        # set params for encoder and decoder
        en_hidden_layers = self.en_params["hidden_layers"]
        en_filters = self.en_params["filters"]
        en_kernel_sizes = self.en_params["kernel_sizes"]
        en_strides = self.en_params["strides"]
        en_paddings = [int(k / 2) for k in en_kernel_sizes]

        de_hidden_layers = self.de_params["hidden_layers"]
        self.de_filters = self.de_params["filters"]
        de_kernel_sizes = self.de_params["kernel_sizes"]
        de_strides = self.de_params["strides"]
        de_paddings = [int(k / 2) for k in de_kernel_sizes]

        # call build function to construct encoder and decoder
        self._build_encoder(en_filters, en_kernel_sizes, en_strides, en_paddings, en_hidden_layers)
        self._build_decoder(self.de_filters, de_kernel_sizes, de_strides, de_paddings, de_hidden_layers)

    def _build_encoder(self, filters, kernel_sizes, strides, paddings, hidden_layers):
        """ builds a convolutional encoder network
        Params:
            filters (list[int]): size of input/output space
            kernel_sizes (list[int]): specify the height/width of the 2D convolution window
            strides (list[int]): specify the stride of the convolution
            paddings (list[int]): specify how the filter should behave when it hits the edge of the matrix
            hidden_layers (int): number of hidden layers
        """

        # create module list
        modules = []
        in_channels = self.input_channels

        # append first and hidden layers to the list
        for h in range(hidden_layers + 1):
            modules.append(
                tnn.Sequential(
                    tnn.Conv2d(in_channels,
                               out_channels=filters[h],
                               kernel_size=kernel_sizes[h],
                               stride=strides[h],
                               padding=paddings[h]),
                    tnn.BatchNorm2d(filters[h]),
                    tnn.LeakyReLU())
            )
            in_channels = filters[h]

        self.encoder = tnn.Sequential(*modules)

        # TODO: maybe adjust initialising input size of output layers
        # specify output layers
        self.out_mean = tnn.Linear(filters[-1]*4, self.latent_dim)
        self.out_var = tnn.Linear(filters[-1]*4, self.latent_dim)

    def _build_decoder(self, filters, kernel_sizes, strides, paddings, hidden_layers):
        """ builds a convolutional encoder network
        Params:
            filters (list[int]): size of input/output space
            kernel_sizes (list[int]): specify the height/width of the 2D convolution window
            strides (list[int]): specify the stride of the convolution
            paddings (list[int]): specify how the filter should behave when it hits the edge of the matrix
            hidden_layers (int): number of hidden layers
        """

        modules = []

        # compute input size to network depending on latent space
        self.decoder_in = tnn.Linear(int(self.latent_dim / 2), filters[0] * 4)

        # append hidden layers to the module list
        for h in range(hidden_layers):
            modules.append(
                tnn.Sequential(
                    tnn.ConvTranspose2d(filters[h],
                                        filters[h+1],
                                        kernel_size=kernel_sizes[h],
                                        stride=strides[h],
                                        padding=paddings[h],
                                        output_padding=paddings[h]),
                    tnn.BatchNorm2d(filters[h+1]),
                    tnn.LeakyReLU())
            )

        self.decoder = tnn.Sequential(*modules)

        # construct output layer (no activation)
        self.decoder_out = tnn.Sequential(
                            tnn.ConvTranspose2d(filters[-1],
                                                filters[-1],
                                                kernel_size=kernel_sizes[-1],
                                                stride=strides[-1],
                                                padding=paddings[-1],
                                                output_padding=paddings[-1]),
                            tnn.BatchNorm2d(filters[-1]),
                            tnn.LeakyReLU(),
                            tnn.Conv2d(filters[-1],
                                       out_channels=self.channels,
                                       kernel_size=kernel_sizes[-1],
                                       padding=paddings[-1]),
                            tnn.Sigmoid())

    def encode(self, x):
        """ Encodes a sequence of images to mean and variance
        Params:
            x (Tensor) [batch_size, seq_len*in_channels, height, width]: contains input sequence of images
        Returns:
            mean (Tensor) [N, latent_dim]: mean of latent distribution
            logvar (Tensor) [N, latent_dim]: variance of latent distribution
        """
        z = self.encoder(x)
        # flatten the output of convolutional network
        z = torch.flatten(z, start_dim=1)
        mean = self.out_mean(z)
        logvar = self.out_var(z)

        return mean, logvar

    def decode(self, q):
        """ decodes an image from a latent position variable
        Params:
            q (Tensor) [N, latent_dim/2]: latent position variable
        Returns:
            x_hat (Tensor) [N, C, H, W]: reconstructed output image
        """
        x_hat = self.decoder_in(q)
        # reshape to fit for first hidden layer
        x_hat = x_hat.view(-1, self.de_filters[0], 2, 2)
        x_hat = self.decoder(x_hat)
        x_hat = self.decoder_out(x_hat)
        return x_hat

    def sample(self, num_samples):
        """ generative function for sampling
        Params:
            num_samples (int): number of samples to be drawn from latent distribution
        Returns:
            x_hat (Tensor) [num_samples, C, H, W]: reconstructed output images
        """
        z = torch.randn(num_samples, self.latent_dim)
        samples = self.decode(z)
        return samples

    def reparameterize(self, mean, logvar):
        """ gets sample from N(mu, var) with reparameterisation trick
        Params:
            mu (Tensor) [N, latent_dim]: contains encoded mean of input
            logvar (Tensor) [N, latent_dim]: contains variance of input
        Returns:
            z (Tensor) [N, latent_dim]: contains latent encoding for one image
        """
        stddev = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(stddev)
        z = epsilon * stddev + mean
        return z

    def forward(self, x):
        """ encodes input sequence into mean and variance for last image and reconstructs it again
        Params:
            x (Tensor) [N, x]: contains flattened sequence of images
        Returns:
            x_hat (Tensor) [N, y]: contains flattened reconstructed image (last one of sequence)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        q = z[:, :int(self.latent_dim / 2)]
        x_hat = self.decode(q)
        return x_hat