import torch
import torch.nn as tnn
import torch.autograd as tgrad
from utils import helper

"""
Models:
    MLP: Simple Linear Fully Connected Network
    LNN: Lagrangian Neural Network
    HNN: Hamiltonian Neural Network
"""


class DDN(torch.nn.Module):
    """ Base Model for Deep Dynamics Model (all models inherit from this one)
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
        self.hidden_dim = self.latent_dim * 64

        self.linear = tnn.ModuleList([tnn.Linear(self.latent_dim, self.hidden_dim)])
        # hidden layer
        self.linear.extend([tnn.Linear(self.hidden_dim, self.hidden_dim) for h in range(self.hidden_layers)])
        # output layer
        self.linear.append(tnn.Linear(self.hidden_dim, self.latent_dim))

    def forward(self, z):
        """ takes latent state as input and outputs parameters for one step into the the future
        Params:
            z (Tensor) [N, latent_dim]: Tensor containing the latent state parameters
        Returns:
            z (Tensor) [N, latent_dim]: Tensor containing the latent state parameters for one step into the future
        """
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

    def lagrangian(self, q, qdot):
        """ takes latent state parameters as input and outputs scalar value for Lagrangian
        Params:
            q (Tensor) [latent_dim/2]: contains latent position
            qdot (Tensor) [latent_dim/2]: contains latent velocity
        Returns:
            lag (Tensor) [1]: scalar value representing the Lagrangian
        """
        # concatenate state parameters
        # z = torch.cat(
        #     (q, qdot), dim=1)

        z = torch.cat(
             (q, qdot))

        x = self.activation(self.input(z))

        for layer in self.hidden:
             x = self.activation(layer(x))

        lag = self.output(x)

        return lag

    def to_config_space(self, z):
        """ splits the latent encoding in generalised coordinates
        Params:
            z (Tensor) [N, latent_dim]: contains latent state parameters
        Returns:
            q (Tensor) [N, latent_dim/2]: latent encoding of position coordinate
            q_dot (Tensor) [N, latent_dim/2]: latent encoding of velocity coordinate
        """
        q = z[:, :int(self.latent_dim / 2)]
        qdot = z[:, int(self.latent_dim / 2):]

        return q, qdot

    def forward(self, q, qdot):
        """ computes the acceleration by calculating gradients of the neural network
            which represents the lagrangian
        Params:
            q (Tensor) [N, latent_dim/2]: contains position parameters
            qdot (Tensor [N, latent_dim/2]: contains position parameters
        Returns:
            dq_ddt (Tensor) [N, latent_dim/2]: contains the approximations for the acceleration
        """

        # init output torch
        batch_size = q.shape[0]
        q_size = q.shape[1]
        dq_ddt = torch.empty(batch_size, q_size)

        # since autograd.functionals can't work with a whole batch at once
        # one needs to iterate over the whole batch

        for b in range(batch_size):
            q_tmp = q[b]
            qdot_tmp = qdot[b]

            # compute hessian an and jacobian of Lagrange Function with respect to input
            hess = tgrad.functional.hessian(self.lagrangian, (q_tmp, qdot_tmp), create_graph=False)
            jac = tgrad.functional.jacobian(self.lagrangian, (q_tmp, qdot_tmp), create_graph=False)

            # calculating the acceleration according to a transformed Euler-Lagrange Equation
            hess_qdot = hess[1][1]
            hess_qqdot = hess[0][1]
            grad_q = jac[0].permute(1, 0)

            hess_inv = torch.linalg.pinv(hess_qdot)
            third_term = torch.matmul(hess_qqdot, qdot_tmp.unsqueeze(-1))
            brackets = torch.sub(grad_q, third_term)
            dq_ddt[b] = (torch.matmul(hess_inv, brackets)).squeeze(1)

        # lag = self.lagrangian(q, qdot)
        #
        # grad_qdot = torch.autograd.grad(lag, qdot, create_graph=True, retain_graph=True,
        #                                 grad_outputs=torch.ones_like(lag))[0]
        # hess_qdot = torch.autograd.grad(grad_qdot, qdot, create_graph=True, retain_graph=True,
        #                                 grad_outputs=torch.ones_like(lag))[0]
        #
        # grad_q = torch.autograd.grad(lag, q, create_graph=True, retain_graph=True,
        #                              grad_outputs=torch.ones_like(lag))[0]
        # hess_qqdot = torch.autograd.grad(grad_q, qdot, create_graph=True, retain_graph=True,
        #                                  grad_outputs=torch.ones_like(lag))[0]
        #
        # hess_inv = torch.linalg.pinv(hess_qdot).permute(1,0)
        # third_term = torch.mul(hess_qqdot, qdot)
        # brackets = torch.sub(grad_q, third_term)
        # dq_ddt = torch.mul(hess_inv, brackets)

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
        Params:
            z (Tensor) [N, latent_dim]: contains latent state parameters
        Returns:
            q (Tensor) [N, latent_dim/2]: latent encoding of position coordinate
            p (Tensor) [N, latent_dim/2]: latent encoding of momentum coordinate
        """
        q = z[:, :int(self.latent_dim / 2)]
        p = z[:, int(self.latent_dim / 2):]
        return q, p

    def forward(self, q, p):
        """ computes the Hamiltonian from position and momentum input
        Params:
            q (Tensor) [N, latent_dim/2]: contains latent position
            p (Tensor) [N, latent_dim/2]: contains latent momentum
        Returns:
            ham (Tensor) [N, 1]: scalar value representing the Hamiltonian
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
    # TODO: implement Variational Integrator Network for comparison
    # @inproceedings{saemundsson2020variational,
    #   title={Variational integrator networks for physically structured embeddings},
    #   author={Saemundsson, Steindor and Terenin, Alexander and Hofmann, Katja and Deisenroth, Marc},
    #   booktitle={International Conference on Artificial Intelligence and Statistics},
    #   pages={3078--3087},
    #   year={2020},
    #   organization={PMLR}
    # }
    def _build_network(self):
        raise NotImplementedError