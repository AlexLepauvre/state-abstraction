from dataclasses import dataclass, field
import numpy as np
import itertools


@dataclass
class LimitedEnergyTask:
    """
    Configuration object for the sequential decision-making task.

    Attributes
    ----------
    n_trials_per_segment : int
        Number of trials per cost segment.
    tmax : int
        Maximum number of trials (time horizon).
    initial_energy : int
        Initial energy level at the start of the task.
    max_energy : int
        Maximum possible energy level.
    min_energy : int
        Minimum possible energy level.
    costs : list[int]
        Possible cost levels for actions.
    energy : list[int]
        Discrete energy states.
    offers : list[int]
        Possible reward offers available at each trial.
    p_offer : list[float]
        Probability distribution over offers.
    actions : list[int]
        Possible actions (typically 0 = reject, 1 = accept).
    """
    # Structure of the task:
    n_trials_per_segment: int = 4
    tmax: int = 9
    
    # Energy
    initial_energy: int = 3
    max_energy: int = 6
    min_energy = 0
    energy_bonus = 0.5

    # Costs
    C: list[int] = field(default_factory=lambda: [1, 2])

    # Rewards:
    E: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6])
    O: list[int] = field(default_factory=lambda: [100, 200, 225, 250, 275, 300, 400])
    p_offer: list[int] = field(default_factory=lambda: [1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7])

    # Actions:
    A: list[int] = field(default_factory=lambda: [0, 1])

    def build(self):
        """
        Constructs the MDP variables: list of all states in state space, transition probability and reward,
        as well as a various metadata (indices...)
        Returns
        -------
        states : list[tuple]
            List of all states in state space. Each state consists of the values of each variables in state space
        s2i : dict(tuple: int), metadata
            Dictionary mapping each state to its index in the transition and reward matrices
        i2s : dict(int: tuple)
            Dictionary mapping each index in the transition and reward matrices to their corresponding states
        tp : (S, A, S) np.array
            Transition probability from each state to the next given the action, formally
            P(s'|s, a)
        r : (S, A) np.array
            Immediate reward for each state given the action
        states, s2i, i2s, tp, r, dict(tdim=self.tdim, e_dim=self.energy_dim) 
        V : (S,) np.ndarray
            State values for all concrete (state-with-time) indices.
        Q : (S, A) np.ndarray
            State-action values.
        dict(tdim, edim): dict
            Index of the time and energy dimensions in the state tuples
        """
        # Create time axis:
        self.T = list(range(1, self.tmax+1))
        self.states = list(itertools.product(self.E, self.O, self.C, self.C, self.T))  # All possible states in state space
        self.s2i = {s:i for i,s in enumerate(self.states)}
        i2s = {i:s for i,s in enumerate(self.states)}
        S, A = len(self.states), len(self.A)
        # The transition probability matrix maps to each state the probability to land in every other possible state, given the action one has taken
        self.tp = np.zeros([S, A, S])
        self.r = np.zeros([S, A])
        self.tdim, self.edim = 4, 0

        # Loop through each possible state:
        for a in self.A:
            for state in self.states:
                # Extract the variables from the current state:
                e, o, cc, fc, t = state
                i = self.s2i[state]
                # In the terminal state, the immediate reward is a function of the energy:
                if t == self.T[-1]:
                    self.r[i, a] = e * self.energy_bonus
                
                # Transitional probability in terminal state is 0, so continue
                if t == self.T[-1]:
                    continue
                
                # Determine the next states variables values:
                # Current and future cost stay fixed, next t is always t+1:
                cc2 = cc  
                fc2 = fc
                t2 = t+1

                # The next energy is a function of action, current energy level and costs (which depends on t)
                if t <= self.n_trials_per_segment:  # Within the first segment, the cost cc applies
                    cost = cc
                else:
                    cost = fc

                # Reward is equal to the offer if the agent accepts and can afford it
                if a == 1 and e >= cost:
                    self.r[i, a] = o

                # If enough energy to accept reward and accept reward, loose energy
                if a == 1 and e >= cost: 
                    e2 = e-cost
                elif a == 1 and e < cost:  # If not enough energy but accept nonetheless, set energy to 0
                    e2 = 0
                else:  # If participants reject, increase energy by 1 up to max energy
                    e2 = min(e+1, self.max_energy)

                # Finally, offer in the next trials are stochastic:
                # Handling the specific case of trials beyond the horizon:
                if t >= self.n_trials_per_segment * 2:
                    # If we are beyond the segment pairs, cc and fc are unknown and are 
                    # equally likely to take any values
                    for cc2, fc2 in list(itertools.product(self.C, self.C)):
                        for o2, p in zip(self.O, self.p_offer):
                            # The probability is the probability of each offer times 
                            # the probability of each possible costs:
                            j = self.s2i[(e2, o2, cc2, fc2, t2)]
                            self.tp[i, a, j] = p * 1/len(self.C)**2
                else:
                    # The next trial can have any of the possible offers depending on their probability:
                    for o2, p in zip(self.O, self.p_offer):
                        j = self.s2i[(e2, o2, cc2, fc2, t2)]
                        self.tp[i, a, j] = p
    