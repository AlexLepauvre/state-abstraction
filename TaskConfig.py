from dataclasses import dataclass, field
import numpy as np
import pandas as pd
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
    tmax: int = 12
    
    # Energy
    initial_energy: int = 3
    max_energy: int = 6
    energy_bonus: float = 0
    min_energy = 0

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
        self.T = list(range(1, self.tmax+2))
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
                # Special handling of the last trial:
                # Terminal slice: t = T+1
                if t == self.tmax + 1:
                    # No transitions: tp[i, a, :] stays all zeros
                    # Terminal reward equals energy_bonus * remaining energy
                    self.r[i, a] = self.energy_bonus * e
                    continue
                # First segment:
                if t <= self.n_trials_per_segment:
                    for o2, p in zip(self.O, self.p_offer):
                        if a == 1 and e >= cc:
                            j = self.s2i[(e-cc, o2, cc, fc, t+1)]
                            self.tp[i, a, j] = p
                            self.r[i, a] = o
                        elif a == 1 and e < cc:
                            j = self.s2i[(0, o2, cc, fc, t+1)]
                            self.tp[i, a, j] = p
                            self.r[i, a] = 0
                        else:
                            j = self.s2i[(min(e+1, self.max_energy), o2, cc, fc, t+1)]
                            self.tp[i, a, j] = p
                            self.r[i, a] = 0
                # Second segment:
                elif self.n_trials_per_segment < t <= self.n_trials_per_segment * 2:
                    for o2, p in zip(self.O, self.p_offer):
                        if a == 1 and e >= fc:
                            j = self.s2i[(e-fc, o2, cc, fc, t+1)]
                            self.tp[i, a, j] = p
                            self.r[i, a] = o
                        elif a == 1 and e < fc:
                            j = self.s2i[(0, o2, cc, fc, t+1)]
                            self.tp[i, a, j] = p
                            self.r[i, a] = 0
                        else:
                            j = self.s2i[(min(e+1, self.max_energy), o2, cc, fc, t+1)]
                            self.tp[i, a, j] = p
                            self.r[i, a] = 0
                # Third segment:
                else:
                    r = []
                    for o2, p in zip(self.O, self.p_offer):
                        for ffc in self.C:
                            if a == 1 and e >= ffc:
                                j = self.s2i[(e-ffc, o2, cc, fc, t+1)]
                                self.tp[i, a, j] += p * 1/len(self.C)
                                r.append(np.mean(self.O))
                            elif a == 1 and e < ffc:
                                j = self.s2i[(0, o2, cc, fc, t+1)]
                                self.tp[i, a, j] += p * 1/len(self.C)
                                r.append(0)
                            else:
                                j = self.s2i[(min(e+1, self.max_energy), o2, cc, fc, t+1)]
                                self.tp[i, a, j] += p  * 1/len(self.C)
                                r.append(0)
                    self.r[i, a] = np.mean(r)

    def design_matrix(self, 
                      n_trials: int, 
                      n_subjects: int
                      ) -> list[pd.DataFrame]:
        """
        Generates the design matrices of each subjects, with n trials each 

        Parameters
        ----------
        n_trials : int
            Number of trials per subjects
        n_subjects : int
            Number of subjects

        Returns
        -------
        list[pd.DataFrame]
            List of pandas dataframe containing the design for each subject
        """
        # Recursively call function if more than 1 subject:
        if n_subjects > 1:
            subjects_design = []
            for _ in range(n_subjects):
                subjects_design.append(self.design_matrix(n_trials, 1))
            return subjects_design
        
        # Calculate the number of segments:
        if n_trials%self.n_trials_per_segment > 0:
            print("Warning, the number of trials doesn't of trials doesn't fall round with the number of trials per segment" \
                  "We'll round up!")
        if np.ceil(n_trials/self.n_trials_per_segment)%2 > 0:
            print("Warning, the number of trials does not allow for a paired number of segments." \
                  "We'll round up!")
            n_trials += self.n_trials_per_segment

        # Calculate the number of segments: 
        n_segments = np.ceil(n_trials/self.n_trials_per_segment)
        # Prepare the possible costs combinations:
        costs_combi = list(itertools.product(self.C, repeat=2))
        design_mat = pd.DataFrame()
        # Loop through each segment
        for seg_pair in range(int(n_segments/2)):
            # Randomly pick one possible cost pair:
            cc, fc = costs_combi[np.random.choice(len(costs_combi), 1)[0]]
            design_mat = pd.concat([design_mat, pd.DataFrame({
                'segment': [seg_pair * 2] * 4 + [seg_pair * 2 + 1] * 4,
                'trial_within_seg': [1, 2, 3, 4] * 2,
                'offer': np.random.choice(self.O, self.n_trials_per_segment * 2),
                'costs': [cc] * 4 + [fc] * 4
            })], ignore_index=True)
        
        return design_mat    
    