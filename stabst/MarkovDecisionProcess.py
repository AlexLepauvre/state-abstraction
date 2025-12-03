import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
from ot import emd2
from joblib import Parallel, delayed
from sklearn.cluster import AgglomerativeClustering
from .utils import draw_sign_boundary, state_classes_from_lbl, avg_reduce_mdp

DISTANCE_METRICS = ['bisimulation', 'q_distance']

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
            title: Optional[str]=None,
            tmax: Optional[float]=None,
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
        if tmax is None:
            tmax = self.T[-2]
        if title is not None:
            fig.suptitle(title, size=14)
        for o_i, o in enumerate(states_var[1]):
            ctr = 0
            for cc in states_var[2]:
                for fc in states_var[3]:
                    mat = np.zeros([len(states_var[self.edim]), tmax])
                    for i, e in enumerate(states_var[self.edim]):
                        for ii, t in enumerate(range(tmax)):
                            mat[i, ii] = DV[self.s2i[(e, o, cc, fc, t+1)]] 
                    
                    im = ax[ctr, o_i].imshow(mat, aspect='auto',
                                            cmap='seismic', origin='lower', vmin=-(np.max(np.abs(DV))), 
                                            vmax=np.max(np.abs(DV)))
                    # Plot the contours:
                    # Draw contour line where Z == 0 (boundary between + and -)
                    draw_sign_boundary(ax[ctr, o_i], mat, thresh=0)
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
    

    def state_pairs_dist(
            self,
            pair_tp: list[list[list[float]]],
            c: list[np.array],
            r_diff: np.array,
            cR: float,
            cT: float
            ) -> float:
        """
        Computes pairwise bisimulation distances as defined by Ferns et al., which is the max 
        of the sum of the absolute difference in immediate reward and earth mover dstance between 
        the transitional probability of two states. 

        Parameters
        ----------
        pair_tp : list[list[float]]
            Transition probability of each state for each action:
            [
                [
                    [Transition probability from s1 given action 1],
                    [Transition probability from s1 given action 2],
                ],
                [
                    [Transition probability from s2 given action 1],
                    [Transition probability from s2 given action 2],
                ]
            ]
        c : list[np.array]
            Cost matrix, contains the costs of moving each bit of one distribution on any other bit 
            of the other distribution for each action. For each action, must be a matrix of length 
            matching the length of the transition probability of s1 and s2 for  that particular action,
            as there must be a cost for moving each units of tp from one state onto the next to constrain
            the coupling. 
            [
                np.array(|s1 Future states|, |s2 Future states|),
                np.array(|s1 Future states|, |s2 Future states|)
            ]

        r_diff : np.array (A)
            Difference in immediate reward between each state for each action
        cR : float
            Weighting of immediate reward in the distance measure
        cT : float
            Weighting of transition (i.e. future reward) in the distance measure

        Returns
        -------
        float
            Distance between two compared states
        """
        # Get actions:
        best = 0
        # Loop over actions:
        for a in range(self.r.shape[1]):
            # Compute distance
            val = cR * r_diff[a] + cT * emd2(pair_tp[0][a], 
                                            pair_tp[1][a], 
                                            c[a])
            # Actualize best value:
            if val > best:
                best = val
        return best


    def bisim_metric(
            self,
            gamma: float = 0.9,
            tol: float = 1e-6,
            max_iters: int = 200,
            njobs=-1
            ) -> np.ndarray:
        """
        Computes Ferns bisimulation distance metric. For a given MDP defined by transitional probabilities between states tp,
        immediate reward 'rewards' and discout factor gamma, this algorithm returns the distance between each pair of states
        within the state space. Bisimulation distance is defined as the sum of the absolute difference in reward and the earth 
        mover distance between the transition probability of two states. This can then be used for state abstraction, i.e. aggregation
        of states together based on their distance. 

        Parameters
        ----------
        tp : np.ndarray
            (S, A, S) transition probabilities
        rewards : np.ndarray
            (S, A) reward
        gamma : float, optional
            Discount factor of the MDP
        tol : float, optional
            precision tolerance of the distance estimation
        max_iters : int, optional
            _description_, by default 200

        Returns
        -------
        np.ndarray
            (S, S) state by state distance matrix

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
        """
        if not 0 < gamma < 1:
            raise ValueError("Gamma must be larger than 0 but smaller than 1!")
        
        # Get shapes:
        S, A, S2 = self.tp.shape
        if S2 != S or self.r.shape != (S, A):
            raise ValueError("tp must be (S,A,S) and rewards must be (S,A)")

        # Reward and tp costs constant:
        cR, cT = (1.0 - gamma), gamma

        # Precompute state pairs of only the upper triangle as the matrix is symetric:
        pairs = [(i, j) for i in range(S) for j in range(i+1, S)]

        # Calculate absolute difference in reward:
        dr = np.zeros((S, S, A))
        for s1, s2 in pairs:
            for a in range(A):
                dr[s1, s2, a] = abs(self.r[s1, a] - self.r[s2, a])

        # Make terminals states absorbing states if needed
        tp_clean = self.tp.copy()
        for s in range(S):
            for a in range(A):
                row = tp_clean[s, a, :]
                total = row.sum()
                if total <= 0:
                    row[:] = 0.0
                    row[s] = 1.0
                elif not np.isclose(total, 1.0, atol=1e-12):
                    raise ValueError("tp rows must sum to 1!")

        # Init distance matrix to be all 0:
        m = np.zeros((S, S), dtype=np.float64)

        # For each state, extract index of next possible states (tp>0) and their associated tp
        next_states_ind = [[np.flatnonzero(tp_clean[s, a, :] > 0) for a in range(A)] for s in range(S)]
        non_zero_tp = [[tp_clean[s, a, next_states_ind[s][a]] for a in range(A)] for s in range(S)]

        # Store non-zero tp of each pairs and action in a list
        states_pairs_tp = [
            [
                [non_zero_tp[s1][a] for a in range(A)],
                [non_zero_tp[s2][a] for a in range(A)],
            ]
            for (s1, s2) in pairs
        ]

        # Fixed-point iteration with shared randomness
        for _ in tqdm(range(max_iters)):
            m_old = m.copy()
            # Create list of costs mapping to each pair the 
            # cost matrix associated with each action
            costs_list = [
                [
                    m[np.ix_(next_states_ind[s1][a], next_states_ind[s2][a])]
                    for a in range(A)
                ]
                for (s1, s2) in pairs
            ]

            # Compute distance for each pair:
            pairs_dist = Parallel(n_jobs=njobs)(
                    delayed(self.state_pairs_dist)(
                                            states_pairs_tp[i], 
                                            costs_list[i], 
                                            dr[s1, s2, :], 
                                            cR, cT) 
                                            for i, (s1, s2) in enumerate(pairs)
                )
            # Replace distance with newest values:
            for i, (s1, s2) in enumerate(pairs):
                m[s1, s2] = pairs_dist[i]
                m[s2, s1] = pairs_dist[i]
            # Fill the diagonal:
            np.fill_diagonal(m, 0.0)
            # Check if within tolerance bounds:
            if np.max(np.abs(m - m_old)) < tol:
                break
        return m/np.max(m)
    

    def qdist(
            self, 
            q: Optional[np.ndarray]=None
            ) -> np.ndarray:
        """
        Computes Q distance between states pair based on the approximate Q function abstraction described in https://arxiv.org/pdf/1701.04113, 
        Q distance is defined as:
        $$d(s_1, s_2) = max|Q_{s1, a} - Q_{s2, a}| \forall a $$
        This metric guarantees that the value function accuracy of the abstract MDP is bounded by:
        (2/epsilon)/(1-lambda)^2

        Parameters
        ----------
        q : (S, A) np.ndarray
            State action value function of an MDP, i.e. value of each action in each state. If none, computed using backward induction

        Returns
        -------
        (S, S) np.ndarray
            Pairwise Q distances
        """
        if q is None:
            _, q = self.backward_induction()
        # Extract size:
        S, _ = self.r.shape
        # Prepare distance matrix:
        d = np.zeros((S, S))
        # Create state pairs:
        pairs = [(i, j) for i in range(S) for j in range(i+1, S)]
        for s1, s2 in pairs:
            val = np.max(np.abs(q[s1, :] - q[s2, :]))
            d[s1, s2] = val
            d[s2, s1] = val
        return d/np.max(d)
    

    def distance_reduce_mdp(
            self,
            eps: float,
            distance_matrix: Optional[np.array]=None,
            distance_measure: Optional[str]=None,
            gamma: Optional[float]=None,
            tol: Optional[float]=None,
            ):
        if distance_matrix is None:
            if gamma is None or tol is None:
                raise ValueError("Must specify distance_measure, gamma and tol if design_matrix not given")
            if distance_measure.lower() == "bisimulation":
                distance_matrix = self.bisim_metric(self, gamma=gamma, tol=tol)
            elif distance_measure.lower() == "q_distance":
                distance_matrix = self.qdist(self, gamma=gamma, tol=tol)
            else:
                raise ValueError(f"Distance must be one of these: {DISTANCE_METRICS}")
        # Find states clusters
        states_lbl = AgglomerativeClustering(distance_threshold=eps, 
                                            n_clusters=None, linkage='complete', metric='precomputed').fit(distance_matrix).labels_
        # Create state classes:
        state_classes = state_classes_from_lbl(self.states, states_lbl)
        # Reduce the MDP accordingly:
        statesR, tpR, rR, class_of_state = avg_reduce_mdp(state_classes, self.tp, self.r, self.s2i)
        # Return MDP object:
        return MDP(statesR, tpR, rR), state_classes, class_of_state
    