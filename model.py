import torch

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

from collections import deque


class FixedSizeQueue:
    def __init__(self, max_len):
        self.queue = deque(maxlen=max_len)
        self.total = 0

    def append(self, value):
        if len(self.queue) == self.queue.maxlen:
            oldest = self.queue.popleft()
            self.total -= oldest
        self.queue.append(value)
        self.total += value

    def get_sum(self):
        return self.total


class NeuralNetwork:
    def __init__(
        self,
        num_neurons: int,
        timestep_duration_ms: float = 0.01,
        resting_charge_level: float = -70,
        min_charge_possible: float = -90,
        fire_threshold: float = -55,
        firing_voltage_peak: float = 30,
        recovery_voltage_per_ms: float = -60,
        update_constant: float = 0.01,
    ):
        self.num_neurons = num_neurons
        self.timestep_duration_ms = timestep_duration_ms
        self.resting_charge_level = resting_charge_level
        self.min_charge_possible = min_charge_possible
        self.fire_threshold = fire_threshold
        self.firing_voltage_peak = firing_voltage_peak
        self.recovery_voltage_per_ms = recovery_voltage_per_ms

        self.neuron_charges = torch.ones(num_neurons) * resting_charge_level
        self.connection_strengths = torch.randn(num_neurons, num_neurons)
        self.time_ms_from_firing = torch.zeros(num_neurons)
        self.grace_status = torch.zeros(num_neurons)

        self.update_constant = update_constant

        hist_len = int(10 / timestep_duration_ms)

        self.positive_hist = FixedSizeQueue(hist_len)
        self.negative_hist = FixedSizeQueue(hist_len)

    def process_input(
        self,
        current: torch.Tensor,
    ):
        self.neuron_charges = torch.clamp(
            (self.neuron_charges + current), min=self.min_charge_possible
        )

    def fire(
        self,
    ):
        # [[a,b,c],
        #  [d,e,f],
        #  [g,h,i]]
        #
        # summing along dim =-1 (each row summed) is the input to the neuron on that row index
        # coordinate xy means the connection from x to y
        # hence filtering out a column means that that neuron doesn't fire (no output)
        # but filtering a row means that that neuron is on grace (less input)

        which_ones_fire = (self.neuron_charges > self.fire_threshold) & (
            self.grace_status == 0
        )
        reset_fire_time = ~which_ones_fire

        self.time_ms_from_firing += self.timestep_duration_ms  # add the timestep
        self.time_ms_from_firing *= reset_fire_time  # set fired neurons to zero

        grace_active = self.grace_status > 0  # finds where grace is active
        grace_filter = grace_active.unsqueeze(0).T * torch.ones_like(
            self.connection_strengths
        )  # each row is the input of each neuron, true along the row if grace is active else false
        in_grace = grace_filter == 1

        firing_filter = which_ones_fire * torch.ones_like(self.connection_strengths)

        grace_zeroed_weights = torch.where(
            in_grace & (self.connection_strengths > 0),
            torch.zeros_like(self.connection_strengths),
            self.connection_strengths,
        )  # Where grace is active, turns positive weights to 0, otherwise changes nothing
        # print(grace_zeroed_weights)

        effective_weights = (
            grace_zeroed_weights * firing_filter
        )  # now filtered everywhere
        neuron_inputs = torch.sum(
            effective_weights, dim=-1
        )  # each neuron now takes that in as a raw current
        # print(effective_weights)
        # print(neuron_inputs)

        self.neuron_charges += neuron_inputs

        self.grace_status = torch.clamp(
            (self.grace_status - self.timestep_duration_ms), min=0
        )  # all neurons cooldown on grace, but not below 0

        # neurons under grace, but their value after is still bigger than -70, get 1 grace frame added back
        on_grace_over_neutral = (self.neuron_charges > self.resting_charge_level) & (
            self.grace_status > 0
        )
        add_back = torch.zeros_like(self.grace_status)
        add_back[on_grace_over_neutral] = self.timestep_duration_ms
        self.grace_status += add_back
        # this way, neurons that are over the natural resting level that just got their grace decreased get it increased
        # This makes it so that if they were below -70, they'd finally get the 2s timer
        grace_recharge = torch.zeros_like(self.neuron_charges)
        grace_recharge[on_grace_over_neutral] = (
            self.recovery_voltage_per_ms * self.timestep_duration_ms
        )
        self.neuron_charges += grace_recharge
        # Neurons in grace and over neutral recharge back towards neutral state

        add_to_grace = (
            torch.ones_like(self.neuron_charges) * 2 * which_ones_fire
        )  # These neurons just fired, so their grace timer has to be turned on

        self.grace_status += add_to_grace

        self.neuron_charges = self.neuron_charges.masked_fill(
            which_ones_fire, self.firing_voltage_peak
        )  # neurons that just fired spike to max voltage

        return which_ones_fire

    def ltp_ltd(
        self,
        just_fired,
    ):
        binary_weights = torch.sign(self.connection_strengths)  # full of +- 1

        difference = self.time_ms_from_firing.unsqueeze(
            0
        ) - self.time_ms_from_firing.unsqueeze(1)
        # difference between last fired and now
        formularized = -torch.atan((difference - 20) / 10)
        # difference of over 20 is negative (flips direction) otherwise positive
        # so if diff is over 20s, its ltd, but less than 20s and its ltp
        # negative diff impossible as masks out the non fired ones, always looks at the receiving one for its inputs

        delta_weights = (
            formularized * just_fired * binary_weights * self.update_constant
        )
        # print(delta_weights.sum())
        self.connection_strengths += delta_weights

    def timed_check(
        self,
        raw_current,
    ):
        self.process_input(raw_current)
        fired = self.fire()
        self.ltp_ltd(fired)
        num_pos = fired[-100:-50].sum().item()
        num_neg = fired[-50:].sum().item()

        self.positive_hist.append(num_pos)
        self.negative_hist.append(num_neg)

        return self.positive_hist.get_sum(), self.negative_hist.get_sum()
