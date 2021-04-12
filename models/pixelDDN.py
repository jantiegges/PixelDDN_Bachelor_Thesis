import torch
from utils.model_output import ModelOutput
from utils import integrator
from models import autoencoder
from models import dynamic_net


##################################################################
### OVERALL MODEL FOR LEARNING DYNAMICS FROM PIXEL OBSERVATION ###
##################################################################

class PixelDDN(torch.nn.Module):
    """ Pixel Deep Dynamic Network """

    def __init__(self, ae_model, ddn_model, ae_params, ddn_params, data_params, latent_dim):
        """
        ae_model (string): name of autoencoder model
        ddn_model (string): name of dynamic network model
        ae_params (dict): model parameters for autoencoder
        ddn_params (dict): model parameters for dynamic network
        data_params (dict): properties of data set
        latent_dim (int): size of latent space
        """
        # init params
        super().__init__()
        self.ae_model = ae_model
        self.ddn_model = ddn_model
        self.en_params = ae_params['encoder']
        self.de_params = ae_params['decoder']
        self.ddn_params = ddn_params
        self.data_params = data_params

        self.channels = data_params['channels']
        self.seq_len = data_params['seq_len']
        self.im_size = data_params['im_size']
        self.delta_t = data_params['delta_time']
        self.latent_dim = latent_dim

        # init time step for integration
        if isinstance(self.delta_t, str):
            num, denom = self.delta_t.split('/')
            self.delta_t = float(num) / float(denom)

        # init integrator (only for LNN and HNN)
        if ddn_model != "MLP":
            self.integrator = integrator.Integrator(method=ddn_params['integrator'], delta_t=self.delta_t)

        # init autoencoder
        if ae_model == "LAE":
            self.ae = autoencoder.LAE(self.en_params,
                                      self.de_params,
                                      self.channels,
                                      self.seq_len,
                                      self.im_size,
                                      latent_dim)
        if ae_model == "CAE":
            self.ae = autoencoder.CAE(self.en_params,
                                      self.de_params,
                                      self.channels,
                                      self.seq_len,
                                      self.im_size,
                                      latent_dim)
        if ae_model == "VAE":
            self.ae = autoencoder.VAE(self.en_params,
                                       self.de_params,
                                       self.channels,
                                       self.seq_len,
                                       self.im_size,
                                       latent_dim)
        if ae_model == "CVAE":
            self.ae = autoencoder.CVAE(self.en_params,
                                       self.de_params,
                                       self.channels,
                                       self.seq_len,
                                       self.im_size,
                                       latent_dim)

        # init dynamic network
        if ddn_model == "MLP":
            self.ddn = dynamic_net.MLP(ddn_params, latent_dim)
        if ddn_model == "LNN":
            self.ddn = dynamic_net.LNN(ddn_params, latent_dim)
        if ddn_model == "HNN":
            self.ddn = dynamic_net.HNN(ddn_params, latent_dim)
        if ddn_model == "VIN":
            raise NotImplementedError

    def forward(self, rollout_batch, pred_steps=1, variational=True):
        """ sets forward pass and return prediction from pixel input """
        raise NotImplementedError


class PixelLNN(PixelDDN):
    """ Pixel Lagrangian Neural Network """

    def forward(self, input_seq, pred_steps=1, variational=True, convolutional=True):
        """ sets forward pass and return prediction from pixel input using a Lagrangian Neural Network
        Params:
            input_seq (Tensor) [batch_size, seq_len, channels, height, width]: sequence of input images
            pred_steps (int): number of timesteps to predict in the future
            variational (bool): whether the autoencoder is variational
            convolutional (bool): whether the autoencoder is convolutional
        """
        # init prediction object
        pred_shape = list(input_seq.shape)
        # length of guessed sequence plus the first one
        pred_shape[1] = pred_steps + 1
        pred = ModelOutput(batch_shape=torch.Size(pred_shape))
        pred.set_input(input_seq)

        # concat along channel dimension
        b, s, c, h, w = input_seq.size()

        # reshape input to match autoencoder input layer
        if convolutional:
            input_seq = input_seq.reshape(b, s * c, h, w)
        else:
            input_seq = input_seq.reshape(b, s * c * h * w)

        # get and save latent distribution
        if variational:
                z_mean, z_logvar = self.ae.encode(input_seq)
                z = self.ae.reparameterize(z_mean, z_logvar)
                pred.set_latent(z, z_mean, z_logvar)
        else:
                z = self.ae.encode(input_seq)
                pred.set_latent(z)

        # initial state
        q, qdot = self.ddn.to_config_space(z)
        pred.append_state(x1=q, x2=qdot, lagrangian=True)

        # initial state reconstruction
        x_reconstructed = self.ae.decode(q)
        if convolutional:
            pred.append_reconstruction(x_reconstructed)
        else:
            x_reconstructed = x_reconstructed.reshape([-1, self.channels, self.im_size, self.im_size])
            pred.append_reconstruction(x_reconstructed)

        # for loop predicting future time steps
        for t in range(pred_steps):
            # compute next state
            q, qdot, qddot = self.integrator.step(x1=q, x2=qdot, ddn=self.ddn, hamiltonian=False)
            pred.append_state(x1=q, x2=qdot, lagrangian=True)
            # append acceleration
            pred.append_acc(qddot)
            # append Lagrangian
            pred.append_energy(self.integrator.energy)

            # compute state reconstruction
            x_reconstructed = self.ae.decode(q)
            if convolutional:
                pred.append_reconstruction(x_reconstructed)
            else:
                x_reconstructed = x_reconstructed.reshape([-1, self.channels, self.im_size, self.im_size])
                pred.append_reconstruction(x_reconstructed)

        # since lagrangian and acceleration is always computed for time step in the past, it needs to be computed one
        # more time
        with torch.no_grad():
            batch_size = q.shape[0]
            q_size = q.shape[1]
            last_energy_tmp = torch.empty(batch_size, q_size)

            for b in range(batch_size):
                q_tmp = q[b]
                qdot_tmp = qdot[b]
                last_energy_tmp[b] = self.ddn.lagrangian(q_tmp, qdot_tmp)

            last_energy = last_energy_tmp.detach().cpu().numpy()

        last_qddot = self.ddn(q, qdot)
        pred.append_energy(last_energy)
        pred.append_acc(last_qddot)

        return pred


class PixelHNN(PixelDDN):
    """ Pixel Hamiltonian Neural Network """

    def forward(self, input_seq, pred_steps=1, variational=True, convolutional=True):
        """ sets forward pass and return prediction from pixel input using a Hamiltonian Neural Network
        Params:
            input_seq (Tensor) [batch_size, seq_len, channels, height, width]: sequence of input images
            pred_steps (int): number of time steps to predict in the future
            variational (bool): whether the autoencoder is variational
            convolutional (bool): whether the autoencoder is convolutional
        """

        # init prediction object
        pred_shape = list(input_seq.shape)
        # length of guessed sequence plus the first one
        pred_shape[1] = pred_steps + 1
        pred = ModelOutput(batch_shape=torch.Size(pred_shape))
        pred.set_input(input_seq)

        # concat along channel dimension
        b, s, c, h, w = input_seq.size()

        # reshape input to match autoencoder input layer
        if convolutional:
            input_seq = input_seq.reshape(b, s * c, h, w)
        else:
            input_seq = input_seq.reshape(b, s * c * h * w)

        # get and save latent distribution
        if variational:
                z_mean, z_logvar = self.ae.encode(input_seq)
                z = self.ae.reparameterize(z_mean, z_logvar)
                pred.set_latent(z, z_mean, z_logvar)
        else:
                z = self.ae.encode(input_seq)
                pred.set_latent(z)

        # initial state
        q, p = self.ddn.to_phase_space(z)
        pred.append_state(x1=q, x2=p, hamiltonian=True)

        # initial state reconstruction
        x_reconstructed = self.ae.decode(q)
        if convolutional:
            pred.append_reconstruction(x_reconstructed)
        else:
            x_reconstructed = x_reconstructed.reshape([-1, self.channels, self.im_size, self.im_size])
            pred.append_reconstruction(x_reconstructed)

        # estimate predictions
        for t in range(pred_steps):
            # compute next state
            q, p = self.integrator.step(x1=q, x2=p, ddn=self.ddn, hamiltonian=True)
            pred.append_state(x1=q, x2=p, hamiltonian=True)
            pred.append_energy(self.integrator.energy)

            # compute state reconstruction
            x_reconstructed = self.ae.decode(q)
            if convolutional:
                pred.append_reconstruction(x_reconstructed)
            else:
                x_reconstructed = x_reconstructed.reshape([-1, self.channels, self.im_size, self.im_size])
                pred.append_reconstruction(x_reconstructed)

        # since energy is always computed for the last timestep, it needs to be computed one more time
        with torch.no_grad():
            last_energy = self.ddn(q, p).detach().cpu().numpy()
        pred.append_energy(last_energy)

        return pred


class PixelMLP(PixelDDN):
    """ Pixel Multilayer Perceptron Network """

    def forward(self, input_seq, pred_steps=1, variational=True, convolutional=True):
        """ sets forward pass and return prediction from pixel input using a Multilayer Perceptron Neural Network
        Params:
            input_seq (Tensor) [batch_size, seq_len, channels, height, width]: sequence of input images
            pred_steps (int): number of timesteps to predict in the future
            variational (bool): whether the autoencoder is variational
            convolutional (bool): whether the autoencoder is convolutional
        """
        # init prediction object
        pred_shape = list(input_seq.shape)
        # length of guessed sequence plus the first one
        pred_shape[1] = pred_steps + 1
        pred = ModelOutput(batch_shape=torch.Size(pred_shape))
        pred.set_input(input_seq)

        # concat along channel dimension
        b, s, c, h, w = input_seq.size()

        # reshape input to match autoencoder input layer
        if convolutional:
            input_seq = input_seq.reshape(b, s * c, h, w)
        else:
            input_seq = input_seq.reshape(b, s * c * h * w)

        # get and save latent distribution
        if variational:
                z_mean, z_logvar = self.ae.encode(input_seq)
                z = self.ae.reparameterize(z_mean, z_logvar)
                pred.set_latent(z, z_mean, z_logvar)
        else:
                z = self.ae.encode(input_seq)
                pred.set_latent(z)

        q = z[:, :int(self.latent_dim / 2)]
        x2 = z[:, int(self.latent_dim / 2):]
        pred.append_state(x1=q, x2=x2)
        x_reconstructed = self.ae.decode(q)

        if convolutional:
            pred.append_reconstruction(x_reconstructed)
        else:
            x_reconstructed = x_reconstructed.reshape([-1, self.channels, self.im_size, self.im_size])
            pred.append_reconstruction(x_reconstructed)

        # estimate predictions
        for t in range(pred_steps):
            # compute state reconstruction
            z = self.ddn(z)
            q = z[:, :int(self.latent_dim / 2)]
            x2 = z[:, int(self.latent_dim / 2):]
            pred.append_state(x1=q, x2=x2)

            x_reconstructed = self.ae.decode(q)

            if convolutional:
                pred.append_reconstruction(x_reconstructed)
            else:
                x_reconstructed = x_reconstructed.reshape([-1, self.channels, self.im_size, self.im_size])
                pred.append_reconstruction(x_reconstructed)

        return pred


class PixelVIN(PixelDDN):
    """ Pixel Variational Integrator Neural Network """

    def forward(self):
        """ sets forward pass and return prediction from pixel input """
        raise NotImplementedError