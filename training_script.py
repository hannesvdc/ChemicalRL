import torch as pt
from torch.optim import Adam

from chemicalrl import MonodLuedekingPiret, SquashedGaussianPolicyNetwork, ODEState

# The ODE model
ode_parameters = {
    "mu_max": 0.40,   # h^-1
    "K_S":    0.10,   # g/L
    "alpha":  0.30,   # gP/gX
    "beta":   0.005,  # gP/gX/h
    "S_f":    50.0,  # g/L (concentrated feed)
    "yield_X": 0.50,  # gX/gS
    "yield_P": 0.20,  # gP/gS
}
ode = MonodLuedekingPiret(ode_parameters)

# Create the policy network
F_max = 5.0
hidden = 64
policy = SquashedGaussianPolicyNetwork(F_max, hidden)

# Reward function with equal weights for now, optimize later.
wP = 1.0
wF = 1.0
def reward(s, F, s_new, dt):
    return wP * ( s_new.P - s.P ) - wF * F * dt

# Sample the initial state. This is a fixed distribution without learnable parameters.
N = 1000
def sampleInitialStates():
    P = pt.zeros(N)
    S = ode_parameters["K_S"] * pt.ones_like(P)
    V = pt.ones_like(P) # V(t = 0) = 1 liter
    
    min_logX = -2
    max_logX = 0
    logX = min_logX + (max_logX-min_logX) * pt.rand_like(P)
    X = pt.exp(logX)
    return ODEState(X, S, P, V, 0.0)

# Time stepping parameters
dt = 0.01
T = 1.0
K = 100
gamma = 0.995 # K \approx 1 / (1 - gamma)
def loss_fn():
    # Build the rewards tensor
    rewards = pt.zeros((N, K))

    # Sample a bunch of initial states
    s = sampleInitialStates()

    # Propagate by sampling actions and evaluating the ODE. Collect the rewards.
    total_log_probs = pt.zeros((N, K))
    for k in range(K):
        log_probs, F = policy.sampleActions(s)
        new_s = ode.evolve(s, F, dt, T)

        # Collect the log-probabilities and rewards
        total_log_probs[:,k] = log_probs
        rewards[:,k] = reward(s, F, new_s, dt)

        # Update the state for the next iteration.
        s = new_s

    # Build the discounted rewards Gt
    Gt = pt.zeros_like(rewards)
    Gt[:,K-1] = rewards[:,K-1]
    for k in range(1, K): # iterate backwards
        Gt[:,K-1-k] = rewards[:,K-1-k] + gamma * Gt[:,K-k]
    
    # Sum discounted rewards with the log-probabilities
    mc_loss = -pt.mean(total_log_probs * Gt.detach())
    return mc_loss

# Build the Adam optimizer with standard paramters
lr = 1.e-3
optimizer = Adam(policy.parameters(), lr)

def _grad_norm(module):
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            total += pt.sum(g * g).item()
    return total ** 0.5

# The main training loop
n_epochs = 1000
for epoch in range(1, n_epochs+1):
    optimizer.zero_grad()

    loss = loss_fn()
    loss.backward()
    optimizer.step()

    _lr = optimizer.param_groups[0]["lr"]
    grad_norm = _grad_norm(policy)
    print(f"[{epoch:04d}] loss={loss.item():.4f}  "
          f"‖g‖={grad_norm:.3e}  lr={_lr:.2e}")