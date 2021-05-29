import torch

# class implementing the numerical integration using the Lagrangian Neural Network or Hamiltonian neural network
# in order to compute future state parameters

"""
Methods:
    Euler: Euler method integration
    RK4: classical Runge-Kutta integration
    Leapfrog: Leapfrog Integration
"""

class Integrator:
    """ Integrator class for different integration methods """

    METHODS = ["Euler", "RK4", "Leapfrog"]

    def __init__(self, delta_t, method="RK4"):
        """
        delta_t (float): time step size between integration steps
        method (string): integration method
        """
        # check for wrong methods
        if method not in self.METHODS:
            print(f"{method} is not a supported integration method")
            raise KeyError

        self.delta_t = delta_t
        self.method = method

    def _get_gradients(self, x1, x2, ddn, remember_energy=False, hamiltonian=False):
        """ compute the gradients of the dynamic input parameters according to Hamiltonian or Lagrangian principles
        Params:
            x1 (Tensor): Latent-space position tensor
            x2 (Tensor): Latent-space velocity/momentum tensor
            ddn: Deep Dynamic Network
            hamiltonian (bool): Whether the x2 is momentum or velocity
            remember_energy (bool): Whether the Hamiltonian/Lagrangian should be saved or not
        Returns:
            dq_dt (tensor): gradient of the position parameter
            dp_dt/dq_ddt (tensor): gradient of the momentum/velocity parameter
        """

        # compute gradients for Hamiltonian formalism
        if hamiltonian:
            # get hamiltonian of the system
            ham = ddn(q=x1, p=x2)

            # dq_dt = dH/dp
            dq_dt = torch.autograd.grad(ham,
                                        x2,
                                        create_graph=True,
                                        retain_graph=True,
                                        grad_outputs=torch.ones_like(ham))[0]

            # dp_dt = -dH/dq
            dp_dt = -torch.autograd.grad(ham,
                                         x1,
                                         create_graph=True,
                                         retain_graph=True,
                                         grad_outputs=torch.ones_like(ham))[0]

            # Returns a new Tensor, detached from the current graph
            if remember_energy:
                self.energy = ham.detach().cpu().numpy()

            return dq_dt, dp_dt

        # lagrangian
        else:
            dq_dt = x2
            # get acceleration
            dq_ddt = ddn(q=x1, qdot=x2)

            # get lagrangian
            if remember_energy:
                batch_size = x1.shape[0]
                q_size = x1.shape[1]
                lag = torch.empty(batch_size, q_size)

                for b in range(batch_size):
                    q_tmp = x1[b]
                    qdot_tmp = x2[b]
                    lag[b] = ddn.lagrangian(q_tmp, qdot_tmp)

                self.energy = lag.detach().cpu().numpy()

            # Returns a new Tensor, detached from the current graph
            # if remember_energy:
            # 	lag = ddn.lagrangian(q=x1, qdot=x2)
            # 	self.energy = lag.detach().cpu().numpy()

            return dq_dt, dq_ddt

    def _euler_step(self, x1, x2, ddn=None, hamiltonian=False):
        """ computes next state with the euler integration method
        Params:
            x1 (Tensor): either position (hamiltonian) or velocity (lagrangian)
            x2 (Tensor): either momentum (hamiltonian) or acceleration (lagrangian)
            ddn: deep dynamic network (lagrangian or hamiltonian)
            hamiltonian (bool): whether dynamic network is hamiltonian or not
        Returns:
            q_next (Tensor): positon parameter of future state
            p_next/qdot_next (Tensor): momentum/velocity parameter of future state
        """
        if hamiltonian:
            q = x1
            p = x2
            dq_dt, dp_dt = self._get_gradients(q, p, ddn, remember_energy=True, hamiltonian=True)

            # Euler integration
            q_next = q + self.delta_t * dq_dt
            p_next = p + self.delta_t * dp_dt
            return q_next, p_next

        # lagrangian
        else:
            q = x1
            qdot = x2
            dq_dt, dq_ddt = self._get_gradients(q, qdot, ddn, remember_energy=True, hamiltonian=False)

            q_next = q + self.delta_t * dq_dt
            qdot_next = qdot + self.delta_t * dq_ddt
            return q_next, qdot_next, dq_ddt

    def _rk4_step(self, x1, x2, ddn=None, hamiltonian=False):
        """ computes next state with the Classic Runge-Kutta integration method
        Params:
            x1 (Tensor): either position (hamiltonian) or velocity (lagrangian)
            x2 (Tensor): either momentum (hamiltonian) or acceleration (lagrangian)
            ddn: deep dynamic network (lagrangian or hamiltonian)
            hamiltonian (bool): whether dynamic network is hamiltonian or not
        Returns:
            q_next (Tensor): positon parameter of future state
            p_next/qdot_next (Tensor): momentum/velocity parameter of future state
        """
        if hamiltonian:
            q = x1
            p = x2

            # k1
            k1_q, k1_p = self._get_gradients(q, p, ddn, remember_energy=True, hamiltonian=True)

            # k2
            q_2 = q + self.delta_t * k1_q / 2  # x = x_t + dt * k1 / 2
            p_2 = p + self.delta_t * k1_p / 2  # x = x_t + dt * k1 / 2
            k2_q, k2_p = self._get_gradients(q_2, p_2, ddn, hamiltonian=True)

            # k3
            q_3 = q + self.delta_t * k2_q / 2  # x = x_t + dt * k2 / 2
            p_3 = p + self.delta_t * k2_p / 2  # x = x_t + dt * k2 / 2
            k3_q, k3_p = self._get_gradients(q_3, p_3, ddn, hamiltonian=True)

            # k4
            q_3 = q + self.delta_t * k3_q / 2  # x = x_t + dt * k3
            p_3 = p + self.delta_t * k3_p / 2  # x = x_t + dt * k3
            k4_q, k4_p = self._get_gradients(q_3, p_3, ddn, hamiltonian=True)

            # Runge-Kutta 4 integration
            q_next = q + self.delta_t * ((k1_q / 6) + (k2_q / 3) + (k3_q / 3) + (k4_q / 6))
            p_next = p + self.delta_t * ((k1_p / 6) + (k2_p / 3) + (k3_p / 3) + (k4_p / 6))

            return q_next, p_next

        # lagrangian
        else:
            q = x1
            qdot = x2

            # k1
            k1_q, k1_qdot = self._get_gradients(q, qdot, ddn, remember_energy=True, hamiltonian=False)

            # k2
            q_2 = q + self.delta_t * k1_q / 2
            qdot_2 = qdot + self.delta_t * k1_qdot / 2
            k2_q, k2_qdot = self._get_gradients(q_2, qdot_2, ddn)

            # k3
            q_3 = q + self.delta_t * k2_q / 2
            qdot_3 = qdot + self.delta_t * k2_qdot / 2
            k3_q, k3_qdot = self._get_gradients(q_3, qdot_3, ddn)

            # k4
            q_4 = q + self.delta_t * k3_q / 2
            qdot_4 = qdot + self.delta_t * k3_qdot / 2
            k4_q, k4_qdot = self._get_gradients(q_4, qdot_4, ddn)

            # Runge-Kutta 4 integration
            q_next = q + self.delta_t * ((k1_q / 6) + (k2_q / 3) + (k3_q / 3) + (k4_q / 6))
            qdot_next = qdot + self.delta_t * ((k1_qdot / 6) + (k2_qdot / 3) + (k3_qdot / 3) + (k4_qdot / 6))

            return q_next, qdot_next, k1_qdot

    def _leapfrog_step(self, x1, x2, ddn=None, hamiltonian=False):
        """ computes next state with the leapfrog integration method
        Params:
            x1 (Tensor): either position (hamiltonian) or velocity (lagrangian)
            x2 (Tensor): either momentum (hamiltonian) or acceleration (lagrangian)
            ddn: deep dynamic network (lagrangian or hamiltonian)
            hamiltonian (bool): whether dynamic network is hamiltonian or not
        Returns:
            q_next (Tensor): positon parameter of future state
            p_next/qdot_next (Tensor): momentum/velocity parameter of future state
        """
        if hamiltonian:
            q = x1
            p = x2
            _, dp_dt = self._get_gradients(q, p, ddn, remember_energy=True, hamiltonian=True)

            # combination of two symplectic euler methods
            # SE1
            p_next_half = p + dp_dt * (self.delta_t/2)
            q_next_half = q + p_next_half * (self.delta_t/2)

            # SE2
            q_next = q_next_half + p_next_half * (self.delta_t/2)
            _, dp_next_dt = self._get_gradients(q_next, p_next_half, ddn, hamiltonian=True)
            p_next = p_next_half + dp_next_dt * (self.delta_t/2)

            return q_next, p_next

        # lagrangian
        else:
            q = x1
            qdot = x2
            _, dq_ddt = self._get_gradients(q, qdot, ddn, remember_energy=True, hamiltonian=False)

            # combination of two symplectic euler methods
            # SE1
            qdot_next_half = qdot + dq_ddt * (self.delta_t/2)
            q_next_half = q + qdot_next_half * (self.delta_t/2)

            # SE2
            q_next = q_next_half + qdot_next_half * (self.delta_t/2)
            _, dq_next_ddt = self._get_gradients(q_next, qdot_next_half, ddn, hamiltonian=False)
            qdot_next = qdot_next_half + dq_next_ddt * (self.delta_t/2)

            return q_next, qdot_next, dq_ddt

    def step(self, x1, x2, ddn, hamiltonian=False):
        """ performs integration for one time step using the specified integration method
        Params:
            x1 (Tensor): either position (hamiltonian) or velocity (lagrangian)
            x2 (Tensor): either momentum (hamiltonian) or acceleration (lagrangian)
            ddn: deep dynamic network (lagrangian or hamiltonian)
            hamiltonian (bool): whether dynamic network is hamiltonian or not
        Returns:
            q_next (Tensor): positon parameter of future state
            p_next/qdot_next (Tensor): momentum/velocity parameter of future state
        """

        if self.method == "Euler":
            return self._euler_step(x1, x2, ddn, hamiltonian)
        elif self.method == "RK4":
            return self._rk4_step(x1, x2, ddn, hamiltonian)
        elif self.method == "Leapfrog":
            return self._leapfrog_step(x1, x2, ddn, hamiltonian)
        else:
            raise NotImplementedError