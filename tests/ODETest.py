import torch as pt
import matplotlib.pyplot as plt

from chemicalrl import ODEState, MonodLuedekingPiret

# Test the ODE evolution for a fixed feeding rate F(t) = F.
# The volume V(t) should grow linearly V(t) = V0 + Ft
# There should also be a consistent dilution .
# Other terms P, S, and X should evolve continuously.

# This function mainly serves to choose a good time-step size dt and integration horizon T (in hours).
def testODE():
    ode_parameters =  {
        "mu_max": 0.40,  # h^-1
        "K_S":    0.10,  # g/L
        "alpha":  0.30,  # gP/gX
        "beta":   0.005, # gP/gX/h
        "S_f":    50.0, # g/L
        "yield_X": 0.50, # gX/gS
        "yield_P": 0.20, # gP/gS
    }
    ode = MonodLuedekingPiret(ode_parameters)

    V0 = 1.0 # 1 liter
    F = 0.1 # 0.1 liters per hour
    S0 = ode_parameters["K_S"]
    X0 = 0.1
    P0 = 0.0
    F_tensor = pt.Tensor([F])
    state = ODEState(pt.Tensor([X0]), pt.Tensor([S0]), pt.Tensor([P0]), pt.Tensor([V0]), 0.0)

    X_history = [X0]
    S_history = [S0]
    P_history = [P0]
    V_history = [V0]
    t_history = [0.0]
    dt = 0.01 # hours
    T = 0.1 # hours
    K = 1000
    for k in range(K):
        print('t =', k * T)
        state = ode.evolve(state, F_tensor, dt, T)
        X_history.append(float(state.X[0]))
        S_history.append(float(state.S[0]))
        P_history.append(float(state.P[0]))
        V_history.append(float(state.V[0]))
        t_history.append(state.t)

    # Plot the evolution of all four state variables
    plt.plot(t_history, X_history, label=r'Biomass Concentration $X(t)$ [cells / liter]')
    plt.plot(t_history, S_history, label=r'Substrate Concentration $S(t)$ [grams / liter]')
    plt.plot(t_history, P_history, label=r'Product Concentration $P(t)$ [grams / liter]')
    plt.plot(t_history, V_history, label=r'Volume $V(t)$ [liters]')
    plt.legend()
    plt.xlabel(r'$t$ [hours]')
    plt.show()

if __name__ == '__main__':
    testODE()