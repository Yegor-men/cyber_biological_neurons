import torch
from torch import nn


class NeuralNetwork:
    def __init__(
        self,
        num_neurons: int,
        base_charge: float = -70,
        fire_level: float = -55,
        min_clamp: float = -90,
        ms_till_decay: float = 20,
        timestep_length_ms: float = 1,
        learning_rate=0.01,
        device: str = "cuda",
    ) -> None:
        self.num_neurons = num_neurons
        self.base_charge = base_charge
        self.fire_level = fire_level
        self.min_clamp = min_clamp

        self.ms_till_decay = ms_till_decay
        # if both neurons fire within decay time: strengthen, 1 in 1 out: weaken, both out: strengthen
        self.timestep_length_ms = timestep_length_ms
        self.decay_per_timestep = timestep_length_ms / ms_till_decay
        # how much decay should happen at each timestep for the input and output monitoring
        self.learning_rate = learning_rate

        self.neuron_charges = torch.ones(num_neurons).to(device) * base_charge
        # charges for the neurons
        self.weights = torch.randn(num_neurons, num_neurons).to(device)
        self.weights.fill_diagonal_(0)
        # a neuron cannot trigger firing itself
        self.input_monitoring = torch.zeros_like(self.weights).to(device)
        self.output_monitoring = torch.zeros_like(self.weights).to(device)
        # these two things try amplify the inputs and the outputs for each neuron when firing
        self.last_fired = torch.zeros_like(self.neuron_charges)

        self.device = device

        self.R = 0
        # positive reward = good, negative = bad

    def update_weights(
        self,
    ):
        updater = self.input_monitoring * self.output_monitoring
        # by how much to update stuff based on time difference etc
        time_difference = self.last_fired.unsqueeze(1) - self.last_fired.unsqueeze(0)
        # diagonal is automatically 0, also A->B and B->A connections are inverse
        delta_weights = updater * time_difference * self.learning_rate * self.R
        self.weights += delta_weights

    def fire(
        self,
        raw_current: torch.Tensor,
    ):
        assert (
            raw_current.shape == self.neuron_charges.shape
        ), f"Raw current should be {self.neuron_charges.shape}"

        device = self.weights.device
        if raw_current.device != device:
            raw_current = raw_current.to(device)

        if self.R != 0:
            self.update_weights()
        # update weights from the last timestep

        self.neuron_charges += raw_current

        ready_to_fire_bool = self.neuron_charges > self.fire_level
        ready_to_fire_float = ready_to_fire_bool.float()
        delta_charge = ready_to_fire_float @ self.weights

        self.neuron_charges = (
            torch.clamp(
                torch.where(
                    self.neuron_charges > self.fire_level,
                    self.base_charge,
                    self.neuron_charges,
                ),
                min=self.min_clamp,
            )
            + delta_charge
        )
        # clamps neuron charges to be between the min range and resets the charge of the fired ones

        self.last_fired = torch.where(
            ready_to_fire_bool, torch.zeros_like(self.last_fired), self.last_fired + 1
        )
        # TODO change this thing from measuring in time steps to use actual time somehow? maybe?

        change = torch.where(
            ready_to_fire_bool,
            torch.ones_like(self.neuron_charges),
            torch.full_like(self.neuron_charges, -self.decay_per_timestep),
        )
        # calculates the decay for the change
        

        self.input_monitoring += change.unsqueeze(0)
        # makes the column (inputs) to be edited
        self.output_monitoring += change.unsqueeze(1)
        # makes the row (outputs) to be edited
        self.input_monitoring /= self.input_monitoring.sum()
        self.output_monitoring /= self.output_monitoring.sum()
        # normalize both of them as not to make values explode
        # repetitive identical updates play less and less of a role


# neurons = 5
# nn = NeuralNetwork(neurons)

# nn.R = 1

# grr = torch.zeros(neurons)
# grr[0] = 30

# print(nn.weights)

# import time

# start = time.time()
# for i in range(10000):
#     nn.fire(grr)
# end = time.time()

# print(nn.weights)


# print(f"{end-start}s taken")
