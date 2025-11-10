import torch as pt
import torch.nn as nn
import math

from chemicalrl import ODEState

class SimpleMLP (nn.Module) :
    def __init__(self, in_dim, out_dim, hidden=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.network(x)
    
class SquashedGaussianPolicyNetwork(pt.nn.Module):
    def __init__(self, F_max, hidden=64):
        super().__init__()
        self.F_max = F_max

        self.backbone = SimpleMLP(4, hidden, hidden)
        self.mu_head = nn.Linear(hidden, 1)
        self.logstd_head = nn.Linear(hidden, 1)

    def forward(self, s):
        h = self.backbone(s)
        mu = self.mu_head(h)
        logstd = self.logstd_head(h)

        return mu, pt.exp(logstd)
    
    def sampleActions(self, s : ODEState):
        mu, sigma = self.forward(s.toTensor())
        mu = pt.flatten(mu)
        sigma = pt.flatten(sigma)
        eps = pt.randn_like(mu)
        u = mu + eps * sigma

        # Compute the action from the random sample
        a = pt.sigmoid(u)
        F = self.F_max * a
        
        # Also compute the log-probability of F
        normal_logp = -0.5*( (u - mu)**2 / sigma**2 + 2.0*pt.log(sigma) + math.log(2.0*math.pi))
        jacobian = -pt.log( a * (1.0 - a))
        log_probs = normal_logp + jacobian - math.log(self.F_max)

        # Return both
        return log_probs, F