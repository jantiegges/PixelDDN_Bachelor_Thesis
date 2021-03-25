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
        autoencoder: autoencoder network
        dynamics: dynamic network
        integrator: TODO
        channels (int): number of channels of the input images (RGB: 3, BW: 1)
        seq_len (int): number of images in one sequence
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
        if isinstance(self.delta_t, str):
            num, denom = self.delta_t.split('/')
            self.delta_t = float(num) / float(denom)
        self.latent_dim = latent_dim

        # init integrator 
        if ddn_model != "MLP":
            self.integrator = integrator.Integrator(method=ddn_params['integrator'], delta_t=self.delta_t)


        # init models
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

    def load(self):
        """ load network parameters """
        raise NotImplementedError

    def save(self):
        """ save network parameters"""
        raise NotImplementedError


class PixelLNN(PixelDDN):
    """ Pixel Lagrangian Neural Network """

    def forward(self, rollout_batch, pred_steps=1, variational=True, convolutional=True):
        """ sets forward pass and return prediction from pixel input """
        # init prediction object
        pred_shape = list(rollout_batch.shape)
        # length of guessed sequence plus the first one
        pred_shape[1] = pred_steps + 1
        pred = ModelOutput(batch_shape=torch.Size(pred_shape))
        pred.set_input(rollout_batch)

        # concat along channel dimension
        b, s, c, h, w = rollout_batch.size()

        if convolutional:
            rollout_batch = rollout_batch.reshape(b, s * c, h, w)
        else:
            rollout_batch = rollout_batch.reshape(b, s * c * h * w)

        # get and save latent distribution
        if variational:
                z_mean, z_logvar = self.ae.encode(rollout_batch)
                z = self.ae.reparameterize(z_mean, z_logvar)
                pred.set_latent(z, z_mean, z_logvar)
        else:
                z = self.ae.encode(rollout_batch)
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

        # estimate predictions
        for t in range(pred_steps):
            # compute next state
            q, qdot = self.integrator.step(x1=q, x2=qdot, ddn=self.ddn, hamiltonian=False)
            pred.append_state(x1=q, x2=qdot, lagrangian=True)
            # append Lagrangian
            pred.append_energy(self.integrator.energy)

            # compute state reconstruction
            x_reconstructed = self.ae.decode(q)
            if convolutional:
                pred.append_reconstruction(x_reconstructed)
            else:
                x_reconstructed = x_reconstructed.reshape([-1, self.channels, self.im_size, self.im_size])
                pred.append_reconstruction(x_reconstructed)


        # since lagrangian is always computed for timestep in the past, it needs to be computed one more time
        with torch.no_grad():
            batch_size = q.shape[0]
            q_size = q.shape[1]
            last_energy_tmp = torch.empty(batch_size, q_size)

            for b in range(batch_size):
                q_tmp = q[b]
                qdot_tmp = qdot[b]
                last_energy_tmp[b] = self.ddn.lagrangian(q_tmp, qdot_tmp)

            last_energy = last_energy_tmp.detach().cpu().numpy()

        pred.append_energy(last_energy)

        return pred


class PixelHNN(PixelDDN):
    """ Pixel Hamiltonian Neural Network """

    def forward(self, rollout_batch, pred_steps=1, variational=True, convolutional=True):
        """ sets forward pass and return prediction from pixel input
        Params:
            rollout_batch (tensor N): tensor which contains the batch
            time_steps (int): number of guessed time steps
            variational (bool): whether the autoencoder is variational or not
        """

        # init prediction object
        pred_shape = list(rollout_batch.shape)
        # length of guessed sequence plus the first one
        pred_shape[1] = pred_steps + 1
        pred = ModelOutput(batch_shape=torch.Size(pred_shape))
        pred.set_input(rollout_batch)

        # concat along channel dimension
        b, s, c, h, w = rollout_batch.size()

        if convolutional:
            rollout_batch = rollout_batch.reshape(b, s * c, h, w)
        else:
            rollout_batch = rollout_batch.reshape(b, s * c * h * w)

        # get and save latent distribution
        if variational:
                z_mean, z_logvar = self.ae.encode(rollout_batch)
                z = self.ae.reparameterize(z_mean, z_logvar)
                pred.set_latent(z, z_mean, z_logvar)
        else:
                z = self.ae.encode(rollout_batch)
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
            pred.append_energy(self.integrator.energy)  # energy of the previous timestep

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

    def forward(self, rollout_batch, pred_steps=1, variational=True, convolutional=True):
        """ sets forward pass and return prediction from pixel input """
        # init prediction object
        pred_shape = list(rollout_batch.shape)
        # length of guessed sequence plus the first one
        pred_shape[1] = pred_steps + 1
        pred = ModelOutput(batch_shape=torch.Size(pred_shape))
        pred.set_input(rollout_batch)

        # concat along channel dimension
        b, s, c, h, w = rollout_batch.size()

        if convolutional:
            rollout_batch = rollout_batch.reshape(b, s * c, h, w)
        else:
            rollout_batch = rollout_batch.reshape(b, s * c * h * w)

        # get and save latent distribution
        if variational:
                z_mean, z_logvar = self.ae.encode(rollout_batch)
                z = self.ae.reparameterize(z_mean, z_logvar)
                pred.set_latent(z, z_mean, z_logvar)
        else:
                z = self.ae.encode(rollout_batch)
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


