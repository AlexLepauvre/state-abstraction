import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib

class MDP:
    """
    Performs dynamic programming (value iteration or backward induction)
    for computing state-action values in a finite-horizon decision problem.
    """
    def __init__(self, 
                 states: list[tuple], 
                 tp: np.ndarray, 
                 r: np.ndarray, 
                 *, 
                 s2i: dict[tuple: int]=None,
                 tdim: Optional[int]=4, 
                 edim: Optional[int]=0
                 ):
        """
        Initialize the dynamic programming solver.

        Parameters
        ----------
        states : list[Tuple]
            List of all possible states in state space
        tp : (S, A, S) np.ndarray
            Transition probability from each state to the next given the action
        r : (S, A) np.ndarray
            Immediate reward from state given action
        s2i : dict[tuple: int], default None
            Dict mapping each state to its index in the tp and r matrices. If None, 
            assume that the order of the tp and r match the order of states
        tdim : int
            Index of the time dimension in state tuples
        edim : int
            index of the time dimension in the state tuples
        """
        # Assert inputs dimensions
        assert len(states) == tp.shape[0], (f"tp matrix does not match the number of states: "
                                            f"{len(states)} states but tp.shape[0]={tp.shape[0]}")
        assert tp.shape[0] == tp.shape[-1], (f"tp matrix dimension mismatch: "
                                             f"tp.shape[0]={tp.shape[0]}, tp.shape[2]={tp.shape[2]}")
        assert r.shape[0] == len(states), (f"r matrix does not match the number of states: "
                                           f"{len(states)} states but r.shape[0]={r.shape[0]}")
        # Set task config:
        self.states, self.tp, self.r = states, tp, r
        # Set indices:
        self.s2i = {state: idx for idx, state in enumerate(self.states)} if s2i is None else s2i # Index of each state
        # Set dimension indices:
        self.tdim, self.edim = tdim, edim
        # Create time axis:
        self.T = sorted(list(set([state[self.tdim] for state in self.states])))

    
    def backward_induction(self):
        """
        Finite-horizon backward induction where time is part of the state, and the
        terminal slice is at time T+1 (no explicit states at T+1 required).

        The first backup (from time T) uses a terminal value vector V_Tplus1:
            Q[i,a] = reward[i,a] + sum_{s'} P[i,a,s'] * V_Tplus1[s']
        Subsequent backups (t < T) use the current V as usual.

        Parameters
        ----------
        states : list[sequence]
            State list; each state includes a time component at index `tdim`.
        transition_proba : np.ndarray, shape (S, A, S)
            Transition probabilities P[s, a, s'].
        reward : np.ndarray, shape (S, A)
            Immediate reward r[s, a].
        tdim : int, default -1
            Index of the time component in each state.

        Returns
        -------
        V : (S,) np.ndarray
            State values for all concrete (state-with-time) indices.
        Q : (S, A) np.ndarray
            State-action values.
        """
        # Extract dimensions:
        S = len(self.states)
        A = self.r.shape[1]
        # Initialize V and Q:
        V = np.zeros(S)
        Q = np.zeros((S, A))

        # Loop through time steps:
        for t in reversed(self.T):
            for a in range(A):
                for state in self.states:
                    if state[self.tdim] != t:
                        continue
                    Q[self.s2i[state], a] = self.r[self.s2i[state], a] + np.dot(self.tp[self.s2i[state], a, :], V)
                    V[self.s2i[state]] = np.max(Q[self.s2i[state], :])
        return V, Q

    def greedy_policy(self, 
                      Q: np.ndarray
                      ):
        """
        Greedy policy based on state action value function, action taken to maximize value

        Parameters
        ----------
        Q : (S, A) np.ndarray
            State action value function, returns the value of each action in each state 
            (given the policy used to compute Q)

        Returns
        -------
        π : (S,) np.ndarray
            Policy
        """
        return np.array(Q[:, 1] - Q[:, 0] > 0).astype(int)

    def evaluate_policy(self, 
                        policy: np.ndarray
                        ):
        """
        Compute expected return under a given policy for a finite-horizon MDP.

        Parameters
        ----------
        policy : (S, ) or (S, A) np.ndarray
            policy mapping action to each state
            if deterministic -> shape (S,) with integer actions
            if stochastic -> shape (S,A) with probs summing to 1
        Returns
        -------
        V : (S,) value function under the specified policy
        Q : (S,A) state action value function under the specified policy
        """
        V = np.zeros(len(self.states))
        Q = np.zeros((len(self.states), self.r.shape[1]))

        # backward sweep
        for t in reversed(self.T):
            idx_t = [i for i, s in enumerate(self.states) if s[self.tdim] == t]
            for i in idx_t:
                for a in range(self.r.shape[1]):
                    Q[i,a] = self.r[i,a] + self.tp[i,a,:] @ V
                # deterministic policy
                if policy.ndim == 1:
                    V[i] = Q[i, policy[i]]
                else:
                    V[i] = (policy[i] * Q[i]).sum()
        return V, Q
    
    def expected_return(self, 
                        policy: np.ndarray, 
                        initial_energy: Optional[int]=0, 
                        t0: Optional[int]=1
                        ):
        """
        Compute expected return under a given policy for a finite-horizon MDP.

        Parameters
        ----------
        policy : (S, ) or (S, A) np.ndarray
            policy mapping action to each state
            if deterministic -> shape (S,) with integer actions
            if stochastic -> shape (S,A) with probs summing to 1
        Returns
        -------
        V : (S,) value function under the specified policy
        Q : (S,A) state action value function under the specified policy
        """
        # Compute policy value:
        V, _ = self.evaluate_policy(policy)

        # Determine the possible starting states (any states with t=t0 and energy value = initial_energy)
        start_idx = [i for i,s in enumerate(self.states) if s[self.tdim]==t0 and s[self.edim]==initial_energy]
        # Set uniform probability over all possible starting states:
        mu0 = np.zeros(len(self.states))
        mu0[start_idx] = 1.0 / len(start_idx)
        # Compute expected returns
        return mu0 @ V
    
    def plot_dv(
            self, 
            DV: np.ndarray,
            title: Optional[str]=None
            ) -> tuple[matplotlib.figure.Figure, plt.Axes]:
        """
        Plot the decision values as a function of energy and 
        time, separately for each offer and costs

        Parameters
        ----------
        DV : np.ndarray
            (S)
        title : Optional[str], default None
            Title of the figure

        Returns
        -------
        tuple[matplotlib.figure.Figure, plt.Axes]
            fig and ax
        """
        # Extract the state variables:
        states_var = [sorted(list(set([state[i] 
                                       for state in self.states])))
                                       for i in range(len(self.states[0]))]
        fig, ax = plt.subplots(len(states_var[2]) ** 2, 
                               len(states_var[1]), 
                               figsize=[12, 8])
        if title is not None:
            fig.suptitle(title, size=14)
        for o_i, o in enumerate(states_var[1]):
            ctr = 0
            for cc in states_var[2]:
                for fc in states_var[3]:
                    mat = np.zeros([len(states_var[self.edim]), len(self.T[:-1])])
                    for i, e in enumerate(states_var[self.edim]):
                        for ii, t in enumerate(self.T[:-1]):
                            mat[i, ii] = DV[self.s2i[(e, o, cc, fc, t)]] 
                    im = ax[ctr, o_i].imshow(mat, aspect='auto',
                                            cmap='seismic', origin='lower', vmin=np.min(DV), 
                                            vmax=np.max(DV))
                    # Plot the contours:
                    # Draw contour line where Z == 0 (boundary between + and -)
                    ax[ctr, o_i].contour(np.array(mat > 0).astype(int), levels=[0.5], 
                                            colors='green', antialiased=False, linewidths=2)
                    if ctr == 0:
                        ax[ctr, o_i].set_title(f'Offer = {o}', size=12)
                    if o_i == 0:
                        ax[ctr, o_i].set_ylabel(f'cc={cc}, fc={fc} \n Energy', size=12)
                    else:
                        ax[ctr, o_i].set_yticklabels([])
                    if ctr + 1 == len(states_var[2]) ** 2:
                        ax[ctr, o_i].set_xlabel('Trials', size=12)
                    else:
                        ax[ctr, o_i].set_xticklabels([])
                    ctr += 1
        plt.tight_layout()
        cbar = fig.colorbar(im, ax=ax.ravel().tolist(),fraction=0.025, pad=0.01)
        cbar.ax.set_ylabel('Decision value', size=16)
        return fig, ax
    