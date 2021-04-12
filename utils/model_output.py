import torch

# class defining defining an object containing all model output parameters that are produced during forwarding

class ModelOutput:
    """ Class for maintaining the model output """

    def __init__(self, batch_shape):
        """
        batch_shape (list[int]): list containing the size of the output batch
        """
        self.input = None
        self.z = None
        self.z_mean = None
        self.z_logvar = None
        self.q = []
        self.p = []
        self.qdot = []
        self.qddot = []
        self.reconstruction = torch.empty(batch_shape)
        self.idx = 0
        self.energies = []

    def set_input(self, input_seq):
        """ saves the input to the ground truth observations
        input_seq (Tensor) [batch_size, seq_len, channels, height, width]: sequence of input images
        """
        self.input = input_seq

    def set_latent(self, z, z_mean=None, z_logvar=None):
        """ saves the encoded latent space variables
        Params:
            z (Tensor) [N, latent_dim]: contains latent state parameters
            z_mean (Tensor) [N, latent_dim]: contains encoded latent mean
            z_logvar (Tensor) [N, latent_dim]: contains encoded latent variance
        """
        self.z = z
        self.z_mean = z_mean
        self.z_logvar = z_logvar

    def append_state(self, x1, x2, hamiltonian=False, lagrangian=False):
        """ saves the state variables
        Params:
            x1 (Tensor): either position (hamiltonian) or velocity (lagrangian)
            x2 (Tensor): either momentum (hamiltonian) or acceleration (lagrangian)
            hamiltonian (bool): whether dynamic network is hamiltonian or not
            lagrangian (bool): whether dynamic network is lagrangian or not
        """
        if hamiltonian:
            self.q.append(x1)
            self.p.append(x2)
        elif lagrangian:
            self.q.append(x1)
            self.qdot.append(x2)
        else:
            self.q.append(x1)
            self.qdot.append(x2)
            self.p.append(x2)

    def append_reconstruction(self, reconstruction):
        """ saves guessed reconstruction
        reconstruction (Tensor) [N, C, H, W]: contains reconstructed images
        """
        self.reconstruction[:, self.idx, ...] = reconstruction

        # increment index
        self.idx += 1

    def append_acc(self, qddot):
        """ saves computed acceleration
        qddot (Tensor) [N, latent_dim]: acceleration
        """
        self.qddot.append(qddot)

    def append_energy(self, energy):
        """ saves estimated lagrangian or hamiltonian
        energy (Tensor) [N, 1]: Lagrangian or Hamiltonian
        """
        self.energies.append(energy)