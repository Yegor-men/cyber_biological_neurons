import torch

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


time_step_duration = 1  # how many ms in a single time step
neuron_reset_time = 2  # how many ms for a neuron to return to rest state
rest_neuron_charge = -70  # resting mv voltage of each neuron
fire_neuron_charge = -55  # how many mv for the neuron to fire
neuron_replenish_per_ms = -30  # how much charge/ms neuron replenishes during recovery
ek_lower_bound = -90  # the lowest charge a neuron can possibly have
fire_spike_charge = 30  # to what voltage the neuron's charge spikes to when firing


class NeuronCluster:
    def __init__(
        self,
        num_neurons,
        time_step_duration=1,
        neuron_reset_time=2,
        rest_neuron_charge=-70,
        fire_neuron_charge=-55,
        neuron_replenish_per_ms=-30,
        ek_lower_bound=-90,
        fire_spike_charge=30,
    ):
        self.num_neurons = num_neurons
        self.time_step_duration = time_step_duration
        self.neuron_reset_time = neuron_reset_time
        self.rest_neuron_charge = rest_neuron_charge
        self.fire_neuron_charge = fire_neuron_charge
        self.neuron_replenish_per_ms = neuron_replenish_per_ms
        self.ek_lower_bound = ek_lower_bound
        self.fire_spike_charge = fire_spike_charge

        self.neuron_charges = (
            torch.ones(num_neurons) * rest_neuron_charge
        )  # charge statuses of the neurons
        self.connections = torch.randn(
            num_neurons, num_neurons
        )  # x axis index is the firing neuron, y axis index is the receiving one
        self.time_since_last_activation_ms = torch.zeros(
            num_neurons
        )  # how many ms from the last time activated
        self.grace_status = (
            torch.zeros(num_neurons)
        )  # how many ms of grace freeze is left. -1 means its in the [-70,30] range, 0 is it's free and positive values is how many ms left until grace is over

    def clamp_neurons(
        self,
    ):
        self.neuron_charges = torch.clamp(self.neuron_charges, min=-90)

    def direct_interference(
        self,
        inputs: torch.Tensor,
    ) -> None:
        assert inputs.shape() == (self.num_neurons)

    def forward(
        self,
        inputs,
    ):
        """
        1. directly input the inputs (punishment vs game state)
        2. clamp the neurons
        3. fire
            a. find out which neurons need to be fired
            b. find out the effective output thing
            c. pass on the 
        4. ltp ltd
            a. observe which neurons just fired
            b. look at their inputs respective to them
            c. adjust the parameters based on stuff
        """
        self.direct_interference(inputs) # directly interfere to add the neurons
        self.clamp_neurons() 
        self.fire() # fire off the neurons 

    # def punish(
    #     self,
    # ):
    #     torch.random.seed()
    #     torch.cuda.random.seed_all()
    #     noise = torch.randn(self.num_neurons)
    #     self.neuron_charges += noise
    #     print(self.neuron_charges)
    #     torch.manual_seed(42)
    #     torch.cuda.manual_seed_all(42)

    # def ltp_ltd(
    #     self,
    # ):
    #     pass

    # def fire(
    #     self,
    # ):
    #     should_fire = self.neuron_charges > self.fire_neuron_charge
    #     firing = self.connections * should_fire
    #     new_additives = torch.sum(firing, dim=-1)
    #     self.intake_inputs(new_additives)

    # def intake_inputs(
    #     self,
    #     inputs: torch.Tensor,
    # ):
    #     assert inputs.shape == (self.num_neurons, self.num_neurons), "Intake inputs wrong size"

    # def forward(
    #     self,
    #     inputs,
    # ):
    #     self.intake_inputs()
    #     self.fire()


test = NeuronCluster(3)
