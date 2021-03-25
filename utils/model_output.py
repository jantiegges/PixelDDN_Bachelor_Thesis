import torch

class ModelOutput:
    """ Class for maintaining the model output """

    def __init__(self, batch_shape):
        self.input = None
        self.z = None
        self.z_mean = None
        self.z_logvar = None
        self.q = []
        self.p = []
        self.qdot = []
        #self.reconstruction = []
        self.reconstruction = torch.empty(batch_shape)
        self.idx = 0
        self.energies = []

    def set_input(self, rollout):
        """ saves the input to the ground truth rollout observations """
        self.input = rollout

    def set_latent(self, z, z_mean=None, z_logvar=None):
        """ saves the encoded latent space variables """
        self.z = z
        self.z_mean = z_mean
        self.z_logvar = z_logvar

    def append_state(self, x1, x2, hamiltonian=False, lagrangian=False):
        """ saves the state variables """
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
        """ saves guessed reconstruction """
        #self.reconstruction.append(reconstruction)
        self.reconstruction[:, self.idx, ...] = reconstruction

        self.idx += 1

    def append_energy(self, energy):
        """ saves estimated energy level """
        self.energies.append(energy)