import torch as pt
import math

from dataclasses import dataclass
from typing import Dict

import warnings

@dataclass
class ODEState:
    X : pt.Tensor
    S : pt.Tensor
    P : pt.Tensor
    V : pt.Tensor
    t : float

    def toTensor(self) -> pt.Tensor:
        return pt.stack((self.X, self.S, self.P, self.V), dim=1)

class MonodLuedekingPiret:

    def __init__(self, 
                 parameters : Dict):
        self.parameters = dict(parameters)

    def evolve(self,
            state : ODEState,
            F : pt.Tensor,
            dt : float,
            T : float) -> ODEState:
        
        # Timestepping using RK4
        t = state.t
        n_steps = int(T / dt)
        if math.fabs(T - n_steps * dt) > 1e-12:
            warnings.warn(f"Time Integration horizon {T} is not a multiple of the step size {dt}. Proceeding with int(T / dt) steps.")

        # Convert the state to a tensor
        tensor_state = state.toTensor()
        for n in range(n_steps):
            tensor_state = self.rk4(tensor_state, F, dt)
        
        return ODEState(tensor_state[:,0], tensor_state[:,1], tensor_state[:,2], tensor_state[:,3], t + T)

    def rhs(self, 
            state : pt.Tensor,
            F : pt.Tensor) -> pt.Tensor:
        X = state[:,0]
        S = state[:,1]
        P = state[:,2]
        V = state[:,3]

        mu_S = self.parameters["mu_max"] * S / (self.parameters["K_S"] + S)
        qP_S = self.parameters["alpha"] * mu_S + self.parameters["beta"]

        dVdt = F
        dXdt = mu_S * X - F / V * X
        dSdt = - mu_S * X / self.parameters["yield_X"] - qP_S * X / self.parameters["yield_P"] + F / V * (self.parameters["S_f"] - S)
        dPdt = qP_S * X - F / V * P
        return pt.stack((dXdt, dSdt, dPdt, dVdt), dim=1)
    
    def rk4(self,
            tensor_state : pt.Tensor,
            F : pt.Tensor,
            dt : float) -> pt.Tensor:
        k1 = self.rhs(tensor_state, F)
        k2 = self.rhs(tensor_state + 0.5 * dt * k1, F)
        k3 = self.rhs(tensor_state + 0.5 * dt * k2, F)
        k4 = self.rhs(tensor_state + dt * k3, F)

        return tensor_state + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)