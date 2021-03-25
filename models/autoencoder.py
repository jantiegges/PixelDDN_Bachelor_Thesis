import torch
import torch.nn as tnn
from utils import helper

"""
Models:
    LAE: Simple Linear Fully Connected Autoencoder
    CAE: Convolutional Autoencoder
    VAE: Variational Autoencoder
    CVAE: Convolutional Variational Autoencoder
    CVAE_RES: Convolutional VAE with Residual Blocks in Decoder
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

        # sets input dim to size of the sequence of images and output to single image
        # self.output_dim = im_size ** 2

        # set other params
        self.latent_dim = latent_dim
        # TODO: adjust activation parameter setting
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
    """ Basic fully-connected MLP Autoencoder"""

    def _build_network(self, im_size, channels, seq_len):

        en_in_features = (im_size ** 2) * seq_len * channels
        en_hidden_layers = self.en_params["hidden_layers"]
        en_hid_features = self.en_params["out_features"]
        en_out_features = self.latent_dim

        # TODO: adjust decoder dimensions (now for forecast of whole sequence)
        de_in_features = int(self.latent_dim/2)
        de_hidden_layers = self.de_params["hidden_layers"]
        de_hid_features = self.de_params["out_features"]
        de_out_features = (im_size ** 2) * channels

        self._build_encoder(en_in_features, en_hidden_layers, en_hid_features, en_out_features)
        self._build_decoder(de_in_features, de_hidden_layers, de_hid_features, de_out_features)

    def _build_encoder(self, in_features, hidden_layers, hid_features, out_features):
         # input layer
        self.encoder = tnn.ModuleList([tnn.Linear(in_features, hid_features[0])])
        # hidden layer
        self.encoder.extend([tnn.Linear(hid_features[h], hid_features[h + 1]) for h in range(hidden_layers)])
        # output layer
        self.encoder.append(tnn.Linear(hid_features[-1], out_features))

    def _build_decoder(self, in_features, hidden_layers, hid_features, out_features):
        # input layer
        self.decoder = tnn.ModuleList([tnn.Linear(in_features, hid_features[0])])
        # hidden layer
        self.decoder.extend([tnn.Linear(hid_features[h], hid_features[h + 1]) for h in range(hidden_layers)])
        # output layer
        self.decoder.append(tnn.Linear(hid_features[-1], out_features))

    def encode(self, x):
        for layer in self.encoder:
            x = self.activation(layer(x))
        return x

    def decode(self, z):
        for layer in self.decoder:
            z = self.activation(layer(z))
        return z

    # TODO: Do I need a forward function at all?
    def forward(self, x):
        # TODO: split x in image for p and q
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
            filters (list(int)): size of input/output space
            kernel_sizes (list(int)): specify the height/width of the 2D convolution window
            strides (list(int)): specify the stride of the convolution
            paddings (list(int)): specify how the filter should behave when it hits the edge of the matrix
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
            filters (list(int)): size of input/output space
            kernel_sizes (list(int)): specify the height/width of the 2D convolution window
            strides (list(int)): specify the stride of the convolution
            paddings (list(int)): specify how the filter should behave when it hits the edge of the matrix
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
        """ Encodes a sequence of images to mean and variance
        Params:
            x (tensor): Tensor of size (batch_size, seq_len*in_channels, height, width) containing the frames
        Returns:
            z
        """
        z = self.encoder(x)
        # flatten the output of convolutional network
        z = torch.flatten(z, start_dim=1)
        z = self.out(z)

        return z

    def decode(self, q):
        """ decodes an image from a latent position variable
        Params:
            q (tensor): latent position variable
        Returns:
            res (tensor NxCxHxW): tensor with reconstructed output image
        """
        x_hat = self.decoder_in(q)
        # reshape to fit for first hidden layer
        x_hat = x_hat.view(-1, self.de_filters[0], 2, 2)
        x_hat = self.decoder(x_hat)
        x_hat = self.decoder_out(x_hat)
        return x_hat

    def forward(self, x):
        z = self.encode(x)
        q = z[:, :int(self.latent_dim / 2)]
        x_hat = self.decode(q)
        return x_hat


class VAE(AE):
    """ Variational Autoencoder """

    def _build_network(self, im_size, channels, seq_len):

        en_in_features = (im_size ** 2) * seq_len * channels
        en_hidden_layers = self.en_params["hidden_layers"]
        en_hid_features = self.en_params["out_features"]
        en_out_features = self.latent_dim

        # TODO: adjust decoder input dimensions depending on dynamic network (latent dim or latent dim / 2)
        de_in_features = int(self.latent_dim/2)
        de_hidden_layers = self.de_params["hidden_layers"]
        de_hid_features = self.de_params["out_features"]
        de_out_features = (im_size ** 2) * channels

        self._build_encoder(en_in_features, en_hidden_layers, en_hid_features, en_out_features)
        self._build_decoder(de_in_features, de_hidden_layers, de_hid_features, de_out_features)

    def _build_encoder(self, in_features, hidden_layers, hid_features, out_features):

        # input layer
        self.encoder = tnn.ModuleList([tnn.Linear(in_features, hid_features[0])])
        # hidden layer
        self.encoder.extend([tnn.Linear(hid_features[h], hid_features[h+1]) for h in range(hidden_layers)])
        # output layer
        self.fc_mu = tnn.Linear(hid_features[-1], out_features)
        self.fc_logvar = tnn.Linear(hid_features[-1], out_features)

    def _build_decoder(self, in_features, hidden_layers, hid_features, out_features):

        # input layer
        self.decoder = tnn.ModuleList([tnn.Linear(in_features, hid_features[0])])
        # hidden layer
        self.decoder.extend([tnn.Linear(hid_features[h], hid_features[h+1]) for h in range(hidden_layers)])

        # output layer
        self.output = tnn.Linear(hid_features[-1], out_features)

    def encode(self, x):
        for layer in self.encoder:
            x = self.activation(layer(x))

        # Split the result into mu and var of the latent Gaussian distribution
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar

    def decode(self, z):
        for layer in self.decoder:
            z = self.activation(layer(z))

        x_hat = torch.sigmoid(self.output(z))
        return x_hat

    def reparameterize(self, mean, logvar):
        """ gets sample from N(mu, var) with reparameterization trick """
        stddev = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(stddev)
        z = epsilon * stddev + mean
        return z

    def forward(self, x):
        # TODO: split x in image for p and q
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        q = z[:, :int(self.latent_dim / 2)]
        x_hat = self.decode(q)
        return x_hat, mu, logvar, z



"""
Code adapted from:
    - Hamiltonian Generative Network (Toth et. al.)
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
            filters (list(int)): size of input/output space
            kernel_sizes (list(int)): specify the height/width of the 2D convolution window
            strides (list(int)): specify the stride of the convolution
            paddings (list(int)): specify how the filter should behave when it hits the edge of the matrix
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

        # TODO: how to set correct input size for linear network (output of convolutional)?
        # specify output layers
        self.out_mean = tnn.Linear(filters[-1]*4, self.latent_dim)
        self.out_var = tnn.Linear(filters[-1]*4, self.latent_dim)

    def _build_decoder(self, filters, kernel_sizes, strides, paddings, hidden_layers):
        """ builds a convolutional encoder network
        Params:
            filters (list(int)): size of input/output space
            kernel_sizes (list(int)): specify the height/width of the 2D convolution window
            strides (list(int)): specify the stride of the convolution
            paddings (list(int)): specify how the filter should behave when it hits the edge of the matrix
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
        """ Encodes a sequence of images to mean and variance
        Params:
            x (tensor): Tensor of size (batch_size, seq_len*in_channels, height, width) containing the frames
        Returns:
            mean (tensor): mean of latent distribution
            logvar (tensor): variance of latent distribution
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
            q (tensor): latent position variable
        Returns:
            res (tensor NxCxHxW): tensor with reconstructed output image
        """
        # get size of latent space
        #batch_size, _ = q.size()
        # specify size of output image
        #output = torch.randn(batch_size, self.channels, self.im_size, self.im_size)

        x_hat = self.decoder_in(q)
        # reshape to fit for first hidden layer
        x_hat = x_hat.view(-1, self.de_filters[0], 2, 2)
        x_hat = self.decoder(x_hat)
        #x_hat = self.decoder_out(x_hat, output_size=output.size())
        x_hat = self.decoder_out(x_hat)
        return x_hat

    def sample(self, num_samples):
        """ TODO: wat is happening in the sample function """
        z = torch.randn(num_samples, self.latent_dim)
        samples = self.decode(z)
        return samples

    def reparameterize(self, mean, logvar):
        """ TODO: what is happening in the reparameterize function """
        stddev = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(stddev)
        z = epsilon * stddev + mean
        return z

    def forward(self, x):
        # TODO: split x in image for p and q
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        q = z[:, :int(self.latent_dim / 2)]
        x_hat = self.decode(q)
        return x_hat


#####
### Test Classes
#####

class LAE_Test(AE):
    """ Basic fully-connected MLP Autoencoder"""

    def _build_network(self, im_size, channels, seq_len):


        en_in_features = (im_size ** 2) * seq_len * channels
        en_hidden_layers = self.en_params["hidden_layers"]
        en_hid_features = self.en_params["out_features"]
        en_out_features = self.latent_dim

        # TODO: adjust decoder dimensions (now for forecast of whole sequence)
        de_in_features = self.latent_dim
        de_hidden_layers = self.de_params["hidden_layers"]
        de_hid_features = self.de_params["out_features"]
        de_out_features = (im_size ** 2) * seq_len * channels

        self._build_encoder(en_in_features, en_hidden_layers, en_hid_features, en_out_features)
        self._build_decoder(de_in_features, de_hidden_layers, de_hid_features, de_out_features)

    def _build_encoder(self, in_features, hidden_layers, hid_features, out_features):

        # input layer
        self.encoder = tnn.ModuleList([tnn.Linear(in_features, hid_features[0])])
        # hidden layer
        self.encoder.extend([tnn.Linear(hid_features[h],hid_features[h+1]) for h in range(hidden_layers)])
        # output layer
        self.encoder.append(tnn.Linear(hid_features[-1], out_features))

        """# Fills the input Tensor with a (semi) orthogonal matrix, as described by in "Exact solution
        # to the nonlinear dynamics of learning in deep linear neural networks" - Saxe, A. et al (2013)
        for layer in self.encoder:
            tnn.init.orthogonal_(layer.weight)"""

    def _build_decoder(self, in_features, hidden_layers, hid_features, out_features):

        # input layer
        self.decoder = tnn.ModuleList([tnn.Linear(in_features, hid_features[0])])
        # hidden layer
        self.decoder.extend([tnn.Linear(hid_features[h], hid_features[h + 1]) for h in range(hidden_layers)])
        # output layer
        self.decoder.append(tnn.Linear(hid_features[-1], out_features))

        """# Fills the input Tensor with a (semi) orthogonal matrix, as described by in "Exact solution
        # to the nonlinear dynamics of learning in deep linear neural networks" - Saxe, A. et al (2013)
        for layer in self.decoder:
            tnn.init.orthogonal_(layer.weight)"""

    def encode(self, x):
        hidden_layers = self.en_params["hidden_layers"]
        for layer in self.encoder:
            x = self.activation(layer(x))
        return x

    def decode(self, z):
        for layer in self.decoder:
            z = self.activation(layer(z))
        return z

    # TODO: Do I need a forward function at all?
    def forward(self, x):
        # TODO: split x in image for p and q
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat


class VAE_Test(AE):
    """ VAriational Autoencoder Test Class"""

    def _build_network(self, im_size, channels, seq_len):

        en_in_features = (im_size ** 2) * seq_len * channels
        en_hidden_layers = self.en_params["hidden_layers"]
        en_hid_features = self.en_params["out_features"]
        en_out_features = self.latent_dim

        # TODO: adjust decoder dimensions (now for forecast of whole sequence)
        de_in_features = self.latent_dim
        de_hidden_layers = self.de_params["hidden_layers"]
        de_hid_features = self.de_params["out_features"]
        de_out_features = (im_size ** 2) * channels

        self._build_encoder(en_in_features, en_hidden_layers, en_hid_features, en_out_features)
        self._build_decoder(de_in_features, de_hidden_layers, de_hid_features, de_out_features)

    def _build_encoder(self, in_features, hidden_layers, hid_features, out_features):

        # input layer
        self.encoder = tnn.ModuleList([tnn.Linear(in_features, hid_features[0])])
        # hidden layer
        self.encoder.extend([tnn.Linear(hid_features[h], hid_features[h+1]) for h in range(hidden_layers)])
        # output layer
        #self.encoder.append(tnn.Linear(hid_features[-1], out_features*2))
        self.fc_mu = tnn.Linear(hid_features[-1], out_features)
        self.fc_logvar = tnn.Linear(hid_features[-1], out_features)

        # self.fc1 = tnn.Linear(8192, 400)
        # self.fc12 = tnn.Linear(400,400)
        # self.fc21 = tnn.Linear(400, 2)
        # self.fc22 = tnn.Linear(400, 2)

        # self.encoder = tnn.Sequential(
        #     tnn.Linear(in_features, hid_features[0]),
        #     tnn.ReLU(),
        #     tnn.Linear(hid_features[0], hid_features[1]),
        #     tnn.ReLU(),
        #     tnn.Linear(hid_features[1], out_features * 2)
        # )

    #
    def _build_decoder(self, in_features, hidden_layers, hid_features, out_features):

        # input layer
        self.decoder = tnn.ModuleList([tnn.Linear(in_features, hid_features[0])])
        # hidden layer
        self.decoder.extend([tnn.Linear(hid_features[h], hid_features[h+1]) for h in range(hidden_layers)])

        #self.decoder.append(tnn.Linear(hid_features[-1], out_features))
        # output layer
        self.output = tnn.Linear(hid_features[-1], out_features)

        # self.fc3 = tnn.Linear(2, 400)
        # self.fc34 = tnn.Linear(400, 400)
        # self.fc4 = tnn.Linear(400, 8192)

        # modules = []
        # modules.append(tnn.Linear(in_features, hid_features[0]))
        # for h in range(hidden_layers):
        #     modules.append(tnn.Linear(hid_features[h], hid_features[h + 1]))
        #
        # self.decoder = tnn.Sequential(*modules)
        #
        # self.output = tnn.Linear(hid_features[-1], out_features)

        # self.decoder = tnn.Sequential(
        #     tnn.Linear(in_features, hid_features[0]),
        #     tnn.ReLU(),
        #     tnn.Linear(hid_features[0], hid_features[1]),
        #     tnn.ReLU(),
        #     tnn.Linear(hid_features[1], out_features),
        #     tnn.Sigmoid()
        # )

    def encode(self, x):
        for layer in self.encoder:
            x = self.activation(layer(x))

        # Split the result into mu and var of the latent Gaussian distribution
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        #mu = x[:,:self.latent_dim]
        #logvar = x[:,self.latent_dim:]

        return mu, logvar
        #
        # h1 = self.activation(self.fc1(x))
        # h1 = self.activation(self.fc12(h1))
        # return self.fc21(h1), self.fc22(h1)
        # z = self.encoder(x)
        # mu, logvar = torch.chunk(z, 2, dim=1)
        #
        # return mu, logvar

    def decode(self, z):
        for layer in self.decoder:
            z = self.activation(layer(z))

        x_hat = torch.sigmoid(self.output(z))
        return x_hat

        # x_hat = self.decoder(z)
        # x_hat = self.output(x_hat)
        # return x_hat

        # h3 = self.activation(self.fc3(z))
        # h3 = self.activation(self.fc34(h3))
        # return torch.sigmoid(self.fc4(h3))
        # x_hat = self.decoder(z)
        # return x_hat

    def reparameterize(self, mean, logvar):
        """ gets sample from N(mu, var) with reparameterization trick """
        stddev = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(stddev)
        z = epsilon * stddev + mean
        return z

    def forward(self, x):
        # TODO: split x in image for p and q
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z


class CVAE_Test(AE):
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
            filters (list(int)): size of input/output space
            kernel_sizes (list(int)): specify the height/width of the 2D convolution window
            strides (list(int)): specify the stride of the convolution
            paddings (list(int)): specify how the filter should behave when it hits the edge of the matrix
            hidden_layers (int): number of hidden layers
        """

        # # create module list
        # modules = []
        # in_channels = self.input_channels
        #
        # # append first and hidden layers to the list
        # for h in range(hidden_layers + 1):
        #     modules.append(
        #         tnn.Sequential(
        #             tnn.Conv2d(in_channels,
        #                        out_channels=filters[h],
        #                        kernel_size=kernel_sizes[h],
        #                        stride=strides[h],
        #                        padding=paddings[h]),
        #             tnn.BatchNorm2d(filters[h]),
        #             tnn.LeakyReLU())
        #     )
        #     in_channels = filters[h]
        #
        # self.encoder = tnn.Sequential(*modules)

        # # TODO: how to set correct input size for linear network (output of convolutional)?
        # # specify output layers
        # self.out_mean = tnn.Linear(filters[-2], self.latent_dim)
        # self.out_var = tnn.Linear(filters[-2], self.latent_dim)

        modules = []
        hidden_dims = [32, 64, 128, 256, 512]

        in_channels = self.input_channels

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                tnn.Sequential(
                    tnn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding = 1),
                    tnn.BatchNorm2d(h_dim),
                    tnn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = tnn.Sequential(*modules)
        self.fc_mu = tnn.Linear(hidden_dims[-1]*4, self.latent_dim)
        self.fc_var = tnn.Linear(hidden_dims[-1]*4, self.latent_dim)



    def _build_decoder(self, filters, kernel_sizes, strides, paddings, hidden_layers):
        """ builds a convolutional encoder network
        Params:
            filters (list(int)): size of input/output space
            kernel_sizes (list(int)): specify the height/width of the 2D convolution window
            strides (list(int)): specify the stride of the convolution
            paddings (list(int)): specify how the filter should behave when it hits the edge of the matrix
            hidden_layers (int): number of hidden layers
        """

        # modules = []
        #
        # # compute input size to network depending on latent space
        # self.decoder_in = tnn.Linear(int(self.latent_dim), filters[0] * 4)
        #
        # # append hidden layers to the module list
        # for h in range(hidden_layers):
        #     modules.append(
        #         tnn.Sequential(
        #             tnn.ConvTranspose2d(filters[h],
        #                                 filters[h + 1],
        #                                 kernel_size=kernel_sizes[h],
        #                                 stride=strides[h],
        #                                 padding=paddings[h],
        #                                 output_padding=paddings[h]),
        #             tnn.BatchNorm2d(filters[h]),
        #             tnn.LeakyReLU())
        #     )
        #
        # self.decoder = tnn.Sequential(*modules)
        #
        # # construct output layer (no activation)
        # self.decoder_out = tnn.ConvTranspose2d(filters[-1],
        #                                        out_channels=self.channels,
        #                                        kernel_size=kernel_sizes[-1],
        #                                        padding=paddings[-1])


        # Build Decoder
        modules = []

        hidden_dims = [32, 64, 128, 256, 512]

        self.decoder_input = tnn.Linear(self.latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                tnn.Sequential(
                    tnn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    tnn.BatchNorm2d(hidden_dims[i + 1]),
                    tnn.LeakyReLU())
            )

        self.decoder = tnn.Sequential(*modules)

        self.final_layer = tnn.Sequential(
                            tnn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            tnn.BatchNorm2d(hidden_dims[-1]),
                            tnn.LeakyReLU(),
                            tnn.Conv2d(hidden_dims[-1], out_channels= 1,
                                      kernel_size= 3, padding= 1),
                            tnn.Tanh())

    def encode(self, x):
        """ Encodes a sequence of images to mean and variance
        Params:
            x (tensor): Tensor of size (batch_size, seq_len*in_channels, height, width) containing the frames
        Returns:
            mean (tensor): mean of latent distribution
            logvar (tensor): variance of latent distribution
        """
        # z = self.encoder(x)
        # # flatten the output of convolutional network
        # z = torch.flatten(z, start_dim=1)
        # mean = self.out_mean(z)
        # logvar = self.out_var(z)
        #
        # return mean, logvar

        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        logvar = self.fc_var(result)

        return mu, logvar

    def decode(self, z):
        """ decodes an image from a latent position variable
        Params:
            q (tensor): latent position variable
        Returns:
            res (tensor NxCxHxW): tensor with reconstructed output image
        """
        # # get size of latent space
        # batch_size, _ = z.size()
        # # specify size of output image
        # output = torch.randn(batch_size, self.channels, self.im_size, self.im_size)
        #
        # res = self.decoder_in(z)
        # # reshape to fit for first hidden layer
        # res = res.view(-1, self.de_filters[0], 2, 2)
        # res = self.decoder(res)
        # res = self.decoder_out(res, output_size=output.size())
        # return res

        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result



    def reparameterize(self, mean, logvar):
        """ gets sample from N(mu, var) with reparameterization trick """
        stddev = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(stddev)
        z = epsilon * stddev + mean
        return z

    def forward(self, x):
        # TODO: split x in image for p and q
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z
