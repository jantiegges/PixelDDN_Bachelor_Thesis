import torch
import torch.nn as tnn
import torch.autograd as tgrad
from utils import helper

"""
Models:
    MLP: Simple Linear Fully Connected Network
    LNN: Lagrangian Neural Network
    HNN: Hamiltonian Neural Network
    VIN: Variational Integrator Network
"""


class DDN(torch.nn.Module):
    """ Base Model for Deep Dynamics Model
    Params:
        dnn_params (dict): contains model parameters for the dynamic network
        latent_dim (int): dimension of latent space
    """

    def __init__(self, ddn_params, latent_dim):
        super(DDN, self).__init__()

        # set params
        self.latent_dim = latent_dim
        self.hidden_layers = ddn_params["hidden_layers"]

        # set activation function
        self.activation = helper.choose_activation(ddn_params["activation"])

        self._build_network()

    def _build_network(self):
        """ builds network architecture """
        raise NotImplementedError()

    def forward(self):
        """ Sets forward pass """
        raise NotImplementedError()


class MLP(DDN):
    """ Baseline MLP model for comparison """

    def _build_network(self, **kwargs):
        # set dimension of hidden layers depending on size of latent space
        # TODO: how to set hidden dim of latent network
        self.hidden_dim = self.latent_dim * 64

        self.linear = tnn.ModuleList([tnn.Linear(self.latent_dim, self.hidden_dim)])
        # hidden layer
        self.linear.extend([tnn.Linear(self.hidden_dim, self.hidden_dim) for h in range(self.hidden_layers)])
        # output layer
        self.linear.append(tnn.Linear(self.hidden_dim, self.latent_dim))

    def forward(self, z):
        for layer in self.linear:
            z = self.activation(layer(z))
        return z


class LNN(DDN):
    """ Lagrangian Neural Network """

    def _build_network(self):
        # set dimension of hidden layers depending on size of latent space
        self.hidden_dim = self.latent_dim * 64

        # set layers
        self.input = tnn.Linear(self.latent_dim, self.hidden_dim)
        self.hidden = tnn.ModuleList(modules=[
            tnn.Linear(self.hidden_dim, self.hidden_dim) for h in range(self.hidden_layers)
        ])
        self.output = tnn.Linear(self.hidden_dim, 1)

        # modules = []
        #
        # # append first and hidden layers to the list
        # for h in range(self.hidden_layers):
        #     modules.append(
        #         tnn.Sequential(
        #             tnn.BatchNorm1d(self.hidden_dim),
        #             tnn.Linear(self.hidden_dim, self.hidden_dim)
        #     ))
        #
        # self.hidden = tnn.Sequential(*modules)

    def lagrangian(self, q, qdot):
        z = torch.cat(
            (q, qdot))

        x = self.activation(self.input(z))

        #x = self.activation(self.hidden(x))
        for layer in self.hidden:
             x = self.activation(layer(x))

        lagrangian = self.output(x)

        return lagrangian

    def to_config_space(self, z):
        """ splits the latent encoding in generalized coordinates
        Args:
            z (Tensor): latent encoding of shape (batch_size, channels, ...)
        Returns:
            q (Tensor): latent encoding of position coordinate
            q_dot (Tensor): latent encoding of velocity coordinate
        """
        q = z[:, :int(self.latent_dim / 2)]
        qdot = z[:, int(self.latent_dim / 2):]
        return q, qdot

    def forward(self, q, qdot):
        """ computes the acceleration by calculating gradients of the neural network
            which represents the lagrangian
        Args:
            q (Tensor (Nxq_size)): contains position parameters
            qdot (Tensor (Nxq_size)): contains position parameters
        Returns:
            dq_ddt (Tensor (Nxq_size)): contains the approximations for the acceleration
        """
        # since autograd.functionals can't work with a whole batch at once
        # one needs to iterate over the whole batch

        # init output torch
        batch_size = q.shape[0]
        q_size = q.shape[1]
        dq_ddt = torch.empty(batch_size, q_size)

        # compute over whole batch

        # compute hessian an and jacobian of Lagrange Function with respect to input
        # hess = tgrad.functional.hessian(self.lagrangian, (q, qdot))
        # jac = tgrad.functional.jacobian(self.lagrangian, (q, qdot))
        #
        # # calculating the acceleration according to a transformed Euler-Lagrange Equation
        # first_term = torch.pinverse(hess[1][1])
        # sec_term = jac[0][0]
        # third_term = hess[0][1]
        #
        # dq_ddt_b = first_term * (sec_term - third_term)

        # iterate over batch
        for b in range(batch_size):
            q_tmp = q[b]
            qdot_tmp = qdot[b]

            # compute hessian an and jacobian of Lagrange Function with respect to input
            hess = tgrad.functional.hessian(self.lagrangian, (q_tmp, qdot_tmp))
            jac = tgrad.functional.jacobian(self.lagrangian, (q_tmp, qdot_tmp))

            # calculating the acceleration according to a transformed Euler-Lagrange Equation
            hess_qdot = hess[1][1]
            hess_qqdot = hess[0][1]
            grad_q = jac[0].permute(1, 0)

            hess_inv = torch.pinverse(hess_qdot)
            third_term = torch.matmul(hess_qqdot, qdot_tmp.unsqueeze(-1))
            brackets = torch.sub(grad_q, third_term)
            dq_ddt[b] = (torch.matmul(hess_inv, brackets)).squeeze(1)

            # first_term = torch.pinverse(hess[1][1])
            # sec_term = jac[0][0]
            # third_term = hess[0][1]
            #
            # dq_ddt[b] = first_term * (sec_term - third_term)

        return dq_ddt


class HNN(DDN):
    """ Hamiltonian Neural Network """

    def _build_network(self):
        # set dimension of hidden layers depending on size of latent space
        self.hidden_dim = self.latent_dim * 64

        # set layers
        self.input = tnn.Linear(self.latent_dim, self.hidden_dim)
        self.hidden = tnn.ModuleList(modules=[
            tnn.Linear(self.hidden_dim, self.hidden_dim) for h in range(self.hidden_layers)
        ])
        self.output = tnn.Linear(self.hidden_dim, 1)

    def to_phase_space(self, z):
        """ splits the latent encoding in canonical coordinates
        Args:
            z (Tensor): latent encoding of shape (batch_size, channels, ...)
        Returns:
            q (Tensor): latent encoding of position coordinate
            p (Tensor): latent encoding of momentum coordinate
        """
        q = z[:, :int(self.latent_dim / 2)]
        p = z[:, int(self.latent_dim / 2):]
        return q, p

    def forward(self, q, p):
        """ computes the Hamiltonian from position and momentum input
        Args:
            q (Tensor (Nxq_size)): contains position parameters
            p (Tensor (Nxq_size)): contains momentum parameters
        Returns:
              ham (tensor Nx1): Hamiltonian representing the total energy of the system
        """
        z = torch.cat(
            (q, p),
            dim=1)
        x = self.activation(self.input(z))
        for layer in self.hidden:
            x = self.activation(layer(x))
        ham = self.output(x)
        return ham


class VIN(DDN):
    """ Variational Integrator Network """

    def _build_network(self):
        raise NotImplementedError
