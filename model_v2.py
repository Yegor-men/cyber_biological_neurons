import torch
from torch import nn

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

"""
# Hyperparameters
tau_e   = 0.1      # 100 ms eligibility decay
eta     = 1e-3     # learning rate
dt      = 0.001    # 1 ms timestep

# Initialization
N = num_neurons
W = torch.randn(N, N) * 0.01             # synaptic weights
e = torch.zeros(N, N)                    # eligibility traces

for each time step:
    # 1) Compute spikes
    charges += raw_current
    spikes = (charges > firing_level).float()
    charges[spikes==1] = base_charge

    # 2) Update eligibility for every synapse i→j
    pre  = spikes.view(N, 1)             # shape [N,1]
    post = spikes.view(1, N)             # shape [1,N]
    hebb = pre * post                    # co-firing indicator
    e    = e - (dt/tau_e)*e + hebb       # decay + bump

    # 3) Detect hit/miss and broadcast reward R
    R = 0
    if paddle_hit:   R = +1
    elif paddle_miss:R = -1

    # 4) Apply weight update when R != 0
    if R != 0:
        W += eta * R * e

    # 5) Continue; no explicit memory of past inputs/outputs
"""




class NeuralNetwork(nn.Module):
    def __init__(
        self,
        num_neurons: int,
        base_charge: float = -70,
        firing_level: float = -55,
        min_clamp: float = -90,
    ) -> None:
        super().__init__()

        self.base_charge = base_charge
        self.firing_level = firing_level
        self.min_clamp = min_clamp

        self.layer1 = nn.Linear(
            in_features=num_neurons, out_features=num_neurons, bias=False
        )

        # self.neuron_charges = (torch.ones(num_neurons) * base_charge).to(self.device)
        self.register_buffer("neuron_charges", torch.ones(num_neurons) * base_charge)

    def forward(
        self,
        raw_current: torch.Tensor,
    ) -> torch.Tensor:
        assert (
            raw_current.shape == self.neuron_charges.shape
        ), f"Raw current should be {self.neuron_charges.shape}"
        
        device = next(self.parameters()).device
        if raw_current.device != device:
            raw_current = raw_current.to(device)
        
        self.neuron_charges += raw_current
        should_fire = (self.neuron_charges > self.firing_level).float()
        delta_charge = self.layer1(should_fire)
        self.neuron_charges = (
            torch.clamp(
                torch.where(
                    self.neuron_charges > self.firing_level,
                    self.base_charge,
                    self.neuron_charges,
                ),
                min=self.min_clamp,
            )
            + delta_charge
        )
        return should_fire

num_neurons = 1000
nn = NeuralNetwork(num_neurons).to("cuda")
foo = torch.zeros(num_neurons)
foo[:10] = 30
# print(foo)

import time

start = time.time()


for i in range(1000):
    out = nn(foo)
    # print(out[10:])

end = time.time()

print(f"{end-start}s total taken")

test = torch.zeros(5)
test[2] = 1

foo = test.view(1,5)
bar = test.view(5,1)

combined = (foo + bar).clamp(max=1)

print(combined)
