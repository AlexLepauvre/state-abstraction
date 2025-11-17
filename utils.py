import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from typing import Optional, Sequence, Tuple
import itertools
from tqdm import tqdm
import numpy as np
from ot import emd2
from joblib import Parallel, delayed


def state_pairs_dist_shared(
        s1: int, 
        s2: int, 
        s1_tp: np.ndarray, 
        s2_tp: np.ndarray,
        r_diff: np.ndarray, 
        m: np.ndarray, 
        cT: float, 
        cR: float
        ) -> tuple[float, int, int]:
    
    # Get actions:
    A = r_diff.shape[0]
    vals = []
    # Loop over actions:
    for a in range(A):
        # match min length; uniform ⇒ simple average
        k = min(s1_tp[a].size, s2_tp[a].size)
        # Sort the variables:
        Iis = np.sort(s1_tp[a])[:k]
        Jjs = np.sort(s2_tp[a])[:k]
        # Compute average distance across all next possible states:
        future = m[Iis, Jjs].mean()
        # Append distance
        vals.append(cR * r_diff[a] + cT * future)
    
    return max(vals), s1, s2


def bisim_metric_shared(
    tp: np.ndarray,            # (S, A, S) transition probabilities
    rewards: np.ndarray,       # (S, A)
    gamma: float = 0.9,
    tol: float = 1e-6,
    max_iters: int = 200,
    njobs=-1
) -> np.ndarray:
    """
    Computes Ferns bisimilarity metric in the specific case where the source of stochasticity is
    shared between states. This is applicable in the case where some variables of the MDP are 
    deterministic, while the stochastic part is taken from the same probability distribution
    across all states (i.e. state independent stochasticity). In this case, linear programming
    can be sidetstep in favor of computing an average under shared randomness. 


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
    # Enforce tp and rewards to numpy arrays:
    tp = np.asarray(tp, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64)
    
    # Get shapes:
    S, A, S2 = tp.shape
    if S2 != S or rewards.shape != (S, A):
        raise ValueError("tp must be (S,A,S) and rewards must be (S,A)")

    # Reward and tp costs constant:
    cR, cT = (1.0 - gamma), gamma

    # Precompute state pairs of only the upper triangle as the matrix is symetric:
    pairs = [(i, j) for i in range(S) for j in range(i+1, S)]

    # Calculate absolute difference in reward:
    dr = np.zeros((S, S, A))
    for s1, s2 in pairs:
        for a in range(A):
            dr[s1, s2, a] = abs(rewards[s1, a] - rewards[s2, a])

    # Make terminals absorbing if needed
    tp_clean = tp.copy()
    for s in range(S):
        for a in range(A):
            row = tp_clean[s, a, :]
            total = row.sum()
            if total <= 0:
                row[:] = 0.0
                row[s] = 1.0
            elif not np.isclose(total, 1.0, atol=1e-12):
                raise ValueError("tp rows must sum to 1!")

    # Init distance matrix with rewards absolute difference
    m = np.zeros((S, S), dtype=np.float64)
    for i in range(S):
        for j in range(i + 1, S):
            best = 0.0
            for a in range(A):
                da = abs(rewards[i, a] - rewards[j, a])
                best = max(best, cR * da)
            m[i, j] = best
    m = np.maximum(m, m.T)  # symmetrize; diag already 0

    # Extract future possible state for each state and action:
    non_zeros_tp = [[np.flatnonzero(tp_clean[s, a, :] > 0) for a in range(A)] for s in range(S)]

    # Fixed-point iteration with shared randomness
    for _ in tqdm(range(max_iters)):
        m_old = m
        m_next = m.copy()
        results = Parallel(n_jobs=njobs)(
                delayed(state_pairs_dist_shared)(s1, s2, non_zeros_tp[s1], non_zeros_tp[s2], dr[s1, s2, :], m_old, cT, cR) for (s1, s2) in pairs
            )
        # Convert to square matrix:
        for dst, s1, s2 in results:
            m_next[s1, s2] = dst
        # Symetrize and set diag to 0:
        m_next = np.maximum(m_next, m_next.T)
        np.fill_diagonal(m_next, 0.0)
        # Check if within tolerance bounds:
        if np.max(np.abs(m_next - m_old)) < tol:
            m = m_next
            break
        m = m_next

    return m


def state_pairs_dist(
        A,
        pair_tp, 
        c,
        r_diff: np.ndarray, 
        cT: float, 
        cR: float
        ) -> tuple[float, int, int]:
    
    # Get actions:
    best = 0
    # Loop over actions:
    for a in range(A):
        # Compute distance
        val = cR * r_diff[a] + cT * emd2(pair_tp[0][a], 
                                         pair_tp[1][a], 
                                         c[a])
        # Actualize best value:
        if val > best:
            best = val
    
    return best


def bisim_metric(
    tp: np.ndarray,            # (S, A, S) transition probabilities
    rewards: np.ndarray,       # (S, A)
    gamma: float = 0.9,
    tol: float = 1e-6,
    max_iters: int = 200,
    njobs=-1
) -> np.ndarray:
    """
    Computes Ferns bisimilarity metric in the specific case where the source of stochasticity is
    shared between states. This is applicable in the case where some variables of the MDP are 
    deterministic, while the stochastic part is taken from the same probability distribution
    across all states (i.e. state independent stochasticity). In this case, linear programming
    can be sidetstep in favor of computing an average under shared randomness. 


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
    # Enforce tp and rewards to numpy arrays:
    tp = np.asarray(tp, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64)
    
    # Get shapes:
    S, A, S2 = tp.shape
    if S2 != S or rewards.shape != (S, A):
        raise ValueError("tp must be (S,A,S) and rewards must be (S,A)")

    # Reward and tp costs constant:
    cR, cT = (1.0 - gamma), gamma

    # Precompute state pairs of only the upper triangle as the matrix is symetric:
    pairs = [(i, j) for i in range(S) for j in range(i+1, S)]

    # Calculate absolute difference in reward:
    dr = np.zeros((S, S, A))
    for s1, s2 in pairs:
        for a in range(A):
            dr[s1, s2, a] = abs(rewards[s1, a] - rewards[s2, a])

    # Make terminals absorbing if needed
    tp_clean = tp.copy()
    for s in range(S):
        for a in range(A):
            row = tp_clean[s, a, :]
            total = row.sum()
            if total <= 0:
                row[:] = 0.0
                row[s] = 1.0
            elif not np.isclose(total, 1.0, atol=1e-12):
                raise ValueError("tp rows must sum to 1!")

    # Init distance matrix with rewards absolute difference
    m = np.zeros((S, S), dtype=np.float64)
    for i in range(S):
        for j in range(i + 1, S):
            best = 0.0
            for a in range(A):
                da = abs(rewards[i, a] - rewards[j, a])
                best = max(best, cR * da)
            m[i, j] = best
    m = np.maximum(m, m.T)  # symmetrize; diag already 0

    # For each state, extract index of next possible states (tp>0) and their associated tp
    next_states_ind = [[np.flatnonzero(tp_clean[s, a, :] > 0) for a in range(A)] for s in range(S)]
    non_zero_tp = [[tp_clean[s, a, next_states_ind[s][a]] for a in range(A)] for s in range(S)]

    # Store non-zero tp of each pairs and action in a list
    states_pairs_tp = []
    for s1, s2 in pairs:
        states_pairs_tp.append([[non_zero_tp[s1][a] for a in range(A)], 
                                [non_zero_tp[s2][a] for a in range(A)]])

    # Fixed-point iteration with shared randomness
    for _ in tqdm(range(max_iters)):
        m_old = m.copy()
        # Create list of costs mapping to each pair the 
        # cost matrix associated with each action
        costs_list = [
            [m[np.ix_(next_states_ind[s1][a], next_states_ind[s2][a])] for a in range(A)]
            for (s1, s2) in pairs
        ]
        # Compute distance for each pair:
        pairs_dist = Parallel(n_jobs=njobs)(
                delayed(state_pairs_dist)(A, 
                                          states_pairs_tp[i], 
                                          costs_list[i], 
                                          dr[s1, s2, :], 
                                          cT, cR) 
                                          for i, (s1, s2) in enumerate(pairs)
            )
        # Replace distance with newest values:
        for i, (s1, s2) in enumerate(pairs):
            m[s1, s2] = pairs_dist[i]
        # Check if within tolerance bounds:
        if np.max(np.abs(m - m_old)) < tol:
            # Symetrize and set diag to 0:
            m = np.maximum(m, m.T)
            np.fill_diagonal(m, 0.0)
            break
    # Symetrize and set diag to 0:
    m = np.maximum(m, m.T)
    np.fill_diagonal(m, 0.0)
    return m


def avg_reduce_mdp(
        classes: list[list[tuple[int]]],
        tp: np.ndarray,
        r: np.ndarray,
        s2i: Optional[dict[tuple:int]]=None
        ) -> tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    """
    Collapse an MDP by averaging the transition probability and reward across states within a class
    
    Parameters
    ----------
    classes : list[list[tuple[int]]]
        List of classes, specfying the states belonging to each class
            [
                [(var1, var2...),(var1,var2,...),....] -> 1st class containing N state tuples with n variables each
            ]
        The transition probability and reward from each class is computed by taking the average
        tp and r of all states within the class
    tp : (S, A, S) np.ndarray
        Transition probability of the full MDP, based on which the tp of the reduced MDP is computed
    r : (S, A) np.ndarray
        Reward of the full MDP, based on which the r of the reduced MDP is computed
    s2i : Optional[dict[tuple:int]]=None]
        Index of each state within the tp and r matrices. In case not provided, will assume
        that the order of the states within class match the order in tp and r when unfolded

    Returns
    tuple[list, np.ndarray, np.ndarray, np.ndarray]
        statesR : list
            all states in reduced state space, taking the first state of each 
            class as a representative
        tpR : np.ndarray
            (K, A, K) transition probability of the reduced state space
        rR : np.ndarray
            (K, A) reward of the reduced state space
        class_of_state : np.ndarray
            (S, ) Maps each state to the class it belongs to
    -------
    
    """
    # Handle inputs:
    if s2i is None:
        s2i = {tuple(state): idx 
               for idx, state  in enumerate(list(itertools.chain(*classes)))}
    # Get dimensions:
    S, A, _ = tp.shape 
    K = len(classes)
    
    # Get index of each state within each class
    classes_ind = [[s2i[tuple(state)] 
                    for state in class_] 
                   for class_ in classes]
    
    # Map each original state -> its class
    class_of_state = np.empty(S, dtype=int)
    for k, idxs in enumerate(classes_ind):
        class_of_state[idxs] = k
    
    # One-hot class indicator matrix M: (S, K), used to sum probs into classes
    M = np.zeros((S, K), dtype=float)
    for k, idxs in enumerate(classes_ind):
        M[idxs, k] = 1.0

    # Average rewards within each class
    rR = np.vstack([r[idxs,:].mean(axis=0) for idxs in classes_ind])

    # Average transitions within each class, then pool to classes
    tpR = np.empty((K, A, K), dtype=float)
    for k, idxs in enumerate(classes_ind):
        for a in range(A):
            # mean over members of class k → distribution over original S
            avg_row = tp[idxs, a, :].mean(axis=0)          # (S,)
            # pool S → K
            tpR[k, a, :] = avg_row @ M              # (K,)
    
    # Pick the first class of each class as a representative:
    statesR = [states[0] for states in classes]

    # Instantiate the new reduced MDP object:
    return statesR, tpR, rR, class_of_state


def reduced2full_value(
        class_of_state : np.ndarray, 
        V_reduced : np.ndarray, 
        Q_reduced : np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
    """
    This function broadcasts the values of and state action values 
    from the reduced state space back into the full state space, by
    setting V and Q of each state to the value of its corresponding 
    class

    Parameters
    ----------
    class_of_state : np.ndarray
        (S,) 1D array containing the class index corresponding to each state
    V_reduced : np.ndarray
        (K, ) 1D array containing the value of each class
    Q_reduced : np.ndarray
        (K, A) 2D array containing the state action value for each class and action

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        V_full : np.ndarray
            (S,) 1D array, values for full state space
        Q_full : np.ndarray
            (S, A) 2D array, state action values for full state space
    """
    
    return V_reduced[class_of_state], Q_reduced[class_of_state]


def state_classes_from_lbl(
        states: list[tuple], 
        class_of_states: np.ndarray
        ) -> list[list[tuple]]: 
    """
    Create classes from labels associating each label to its class

    Parameters
    ----------
    states : list[tuple]
        _description_
    class_of_states : np.ndarray
        _description_

    Returns
    -------
    list[list[tuple]]
        _description_
    """
    # Prepare state classes:
    state_classes = [[] for _ in range(len(np.unique(class_of_states)))]
    for state_i, state in enumerate(states):
        state_classes[class_of_states[state_i]].append(state)
    return state_classes

def group_var(
        vals: list
        ) -> list[list]:
    """
    Aggregates values within a list in groups from pairs (i.e. n=2) to
    n = len(vals) (i.e. forming a single group containing all values within a list).
    Importantly, beyond a half split, only consider a group of all values together.
    [1,2,3,4,5,6,7] →
        [[ [1,2], [3,4], [5,6], [7] ],
        [ [1,2,3], [4,5,6], [7] ],
        [ [1,2,3,4], [5,6,7] ],
        [ [1,2,3,4,5,6,7] ]]

    Parameters
    ----------
    vals : list
        List containing the sequence of values to group in increments of n to len(vals)

    Returns
    -------
    groups : list[list]
        List of lists containing the groups at various levels of grouping
    """
    n = len(vals)
    groups = []
    for grp_size in range(2, n + 1):
        if grp_size > n // 2 + 1:  # beyond halfway, only take the full set
            if not groups or groups[-1] != [vals]:
                groups.append([vals])
            break
        groups.append([vals[i:i+grp_size] for i in range(0, n, grp_size)])
    return groups


def aggregate_states(
        states: list[tuple[int]], 
        dim: int
        ) -> list[list[tuple]]:
    """
    Generates classes (i.e. groupings of states) by lumping variables of one dimension of 
    the state space:
    states = [(1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 1, 4)...]
        -> classes = [[(1, 1, 1), (1, 1, 2)], [(1, 1, 3), (1, 1, 4)]...]
    Generates several reduced state spaces by generating groupings of all possible size for the
    required dimension (lumping in pairs, in triplets...)

    Parameters
    ----------
    states : list[tuple[int]]
        List of all possible states in state space
    dim : int
        Dimensions along which to aggregate values

    Returns
    -------
    list[list[list[tuple]]]
        List of all reduced state spaces. Each reduced state space consists itself of a 
        nested list, specfying the states belonging to each class
            [
                [(var1, var2...),(var1,var2,...),...] -> 1st class containing N state tuples with n variables each
            ]
    """

    # Create the groups associated with this dimension:
    groups = group_var(list(sorted(set([state[dim] 
                                        for state in states]))))
    classes = []
    for group_i, group in enumerate(groups):
        classes.append([])
        # Loop through each state:
        for state in states:
            # Loop through each group
            for g in group:
                if state[dim] == g[0]:
                    classes[group_i].append([tuple(list(state[:dim]) + [v] + list(state[dim+1:])) for v in g]) 
                    break

    return classes


def bisimulation_classes(
        states: list[list[int]], 
        tp: np.ndarray,    # shape (S, A, S)
        r: np.ndarray,    # shape (S, A, S)
        verbose: Optional[bool] = False): 
    """
    Identify equivalent states by comparing the transition probability and reward of each pairs of 
    states in states. Returns a list of lists grouping all equivalent states into a class (list of 
    equivalent states)
    
    Parameters
    ----------
    states : list[list[int]]
        List of list containing each state with each state variable for that state
    tp : np.ndarray, shape (S, A, S)
        Transition probabilities P[s, a, s'].
    r : np.ndarray, shape (S, A)
        Immediate reward r[s, a].
    verbose: boolean
        Print info about state reduction
    """
    state_to_idx = {state: idx for idx, state in enumerate(states)}
    idx_to_state = {idx: state for idx, state in enumerate(states)}
    # Compute all possible pairs:
    states_equivalence = np.zeros([len(states), len(states)])

    # Compare all possible pairs:
    for i, state1 in enumerate(states):
        for ii, state2 in enumerate(states):
            # Extract the TP of each state:
            tp1 = tp[state_to_idx[state1], :, :]
            tp2 = tp[state_to_idx[state2], :, :]
            # Extract the reward:
            r1 = r[state_to_idx[state1], :]
            r2 = r[state_to_idx[state2], :]
            if (tp1 == tp2).all() and (r1 == r2).all():
                states_equivalence[i, ii] = 1

    state_classes = []
    for i in range(states_equivalence.shape[0]):
        # Skipping states that are already part of a cluster:
        if idx_to_state[i] in list(itertools.chain.from_iterable(state_classes)):
            continue
        cluster = [idx_to_state[idx] for idx in np.where(np.squeeze(states_equivalence[i, :]) == 1)[0]]
        state_classes.append([idx_to_state[idx] for idx in np.where(np.squeeze(states_equivalence[i, :]) == 1)[0]])

    # We can now print each cluster:
    if verbose:
        for i, cluster in enumerate(state_classes):
            if len(cluster) > 1:
                print(f"States in cluster {i}: ")
                for state in cluster:
                    print(f'[e={state[0]}, o={state[1]}, cc={state[2]}, fc={state[3]}, t={state[4]}]')
    
        # Calculate the proportion of redundant states:
        n_clusters = len([cluster for cluster in state_classes if len(cluster)])
        n_states_in_clusters = len(list(itertools.chain.from_iterable([cluster for cluster in state_classes if len(cluster) > 1])))
        print(f'Results of states clustering: ')
        print(f'     Number of clusters: {n_clusters}')
        print(f'     Number of states in clusters: {n_states_in_clusters} out of {states_equivalence.shape[0]}')
        print(f'     Reduced state space: {n_clusters + states_equivalence.shape[0] - n_states_in_clusters}')
        print(f'     Relative state space reduction: {(len(states) - n_clusters) / len(state)}')

    return state_classes, states_equivalence
    

def plot_state_matrix(
        matrix: np.ndarray,
        states: Sequence[Sequence[int]],
        depth: int = 3,
        cmap: str = "viridis",
        max_labels_per_tier: int = 60,
        rotate_x: bool = True,
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        feature_names: Optional[list] = None,
        cbar_label: Optional[str] = None,

 ) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a state-by-state matrix
    with hierarchical labels. X-axis tiers are placed *below* the heatmap and
    Y-axis tiers to the *left*, with outermost (top-level) labels farthest from
    the heatmap.

    Each tier represents one dimension of the state vector, for example:
    energy, offer, current cost, future cost, or time. The function constructs
    hierarchical labels that visualize how state groups are nested.

    Parameters
    ----------
    matrix : np.ndarray
        2D square array of shape (N, N) representing state-to-state distances
        or correspondences.
    states : Sequence[Sequence[int]]
        Array-like of shape (N, D) where each row represents one state.
        Columns are typically ordered as [E, O, CC, FC, T].
        Must be lexicographically sorted along these dimensions for proper
        block structure.
    depth : int, optional
        Number of hierarchical levels (1–5) to display. The first `depth`
        features in `states` are used. Default is 3.
    cmap : str, optional
        Colormap used for the matrix. Default is ``'viridis'``.
    max_labels_per_tier : int, optional
        Maximum number of tick labels to display per tier; labels are
        automatically subsampled if exceeded. Default is 60.
    rotate_x : bool, optional
        Whether to rotate x-axis tick labels by 45 degrees. Default is True.
    feature_names : list, optional
        Name of each of the dimensions of the state space

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Matplotlib Figure containing the plot.
    ax : matplotlib.axes.Axes
        The main heatmap Axes.

    Notes
    -----
    - Uses `mpl_toolkits.axes_grid1.make_axes_locatable` with fixed physical
      sizes (in mm) to ensure consistent spacing and alignment regardless of
      figure size or layout manager.
    - Works best if `states` is lexicographically sorted according to the
      column order (e.g., [E, O, CC, FC, T]).
    - To make tiers snug to the heatmap, the figure disables
      `constrained_layout` and applies a minimal `tight_layout()`.

    Examples
    --------
    >>> E, O, C, T = range(3), range(2), [1, 2], range(4)
    >>> import itertools
    >>> states = list(itertools.product(E, O, C, C, T))
    >>> matrix = np.random.rand(len(states), len(states))
    >>> fig, ax = plot_state_matrix(matrix, states, depth=3)
    >>> plt.show()
    """

    # =================== Handle inputs ===================
    if cbar_label is None:
        cbar_label = 'Distance'
    # Convert states to a numpy array:
    states = np.array(states)
    # Check dimensions:
    assert states.ndim == 2, 'States should be provided in flatten 2D array'
    N = len(states)
    assert 1 <= depth <= 5
    if feature_names is None:
        feature_names = [f'dim: {i}' for i in range(states.shape[1])]
    else:
        assert len(feature_names) == states.shape[1], 'Feature name size doesnt match states'
    assert 1 <= depth <= 5
    
    # ==== Compute boundaries and centers of each level ====
    centers_per_level, labels_per_level, boundaries_all = [], [], []
    for l in range(1, depth+1):
        vals = states[:, :l]
        uniq, first_idx, counts = np.unique(vals, axis=0, return_index=True, return_counts=True)
        centers = first_idx + counts/2.0
        centers_per_level.append(centers)
        feat = feature_names[l-1]
        this_vals = uniq[:, -1]
        labels = [f"{feat}={int(v)}" for v in this_vals]
        labels_per_level.append(labels)
        boundaries_all.append(first_idx[1:] - 0.5)

    # ====== Prepare grid to place each levels labels ======
    gs = plt.figure(figsize=(15, 15)).add_gridspec(
        nrows=depth + 2, ncols=depth + 2,
        width_ratios=[1]*depth + [30.0, 1],
        height_ratios=[30.0] + [1]*depth + [0.5]  # last row smaller (was 1)
    )

    fig = plt.gcf()

    # ==================== Plot heatmap ====================
    ax = fig.add_subplot(gs[0, depth])
    im = ax.imshow(matrix, cmap=cmap, interpolation='none')
    ax.set_xlim(-0.5, N-0.5)
    ax.set_ylim(N-0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([])

    # =================== Draw boundaries ==================
    for i, bounds in enumerate(boundaries_all):
        for b in bounds:
            ax.axhline(b, color='white', lw=0.05 + 0.1*(depth - i), alpha=0.5)
            ax.axvline(b, color='white', lw=0.05 + 0.1*(depth - i), alpha=0.5)

    def subsample(centers, labels):
        if len(centers) <= max_labels_per_tier: return centers, labels
        step = int(np.ceil(len(centers) / max_labels_per_tier))
        return centers[::step], labels[::step]

    # =================== Plot x labels ==================
    for lvl in range(depth):  # 0 = outermost (farthest)
        row = depth + 1 - lvl
        ax_bottom = fig.add_subplot(gs[row, depth], frame_on=False)
        ax_bottom.set_xlim(-0.5, N-0.5)
        ax_bottom.xaxis.set_ticks_position('top')
        ax_bottom.yaxis.set_visible(False)
        for s in ax_bottom.spines.values(): s.set_visible(False)
        centers, labels = subsample(centers_per_level[lvl], labels_per_level[lvl])
        ax_bottom.set_xticks(centers)
        ax_bottom.set_xticklabels(labels, rotation=45 if rotate_x else 0,
                                  ha='center', va='top', fontsize=8 + 1 * (depth-lvl))
        ax_bottom.tick_params(axis='x', length=0, pad=0)
        pos = ax_bottom.get_position()
        ax_bottom.set_position([pos.x0, pos.y0 + 0.05, pos.width, pos.height])

    # =================== Plot y labels ==================
    for lvl in range(depth):
        ax_left = fig.add_subplot(gs[0, lvl], frame_on=False)
        ax_left.set_ylim(N-0.5, -0.5)
        ax_left.set_xlim(0, 1)
        ax_left.yaxis.set_ticks_position('right')
        ax_left.xaxis.set_visible(False)
        for s in ax_left.spines.values(): s.set_visible(False)
        centers, labels = subsample(centers_per_level[lvl], labels_per_level[lvl])
        ax_left.set_yticks(centers)
        ax_left.set_yticklabels(labels, fontsize=8 + 1 * (depth-lvl), rotation=45)
        ax_left.tick_params(axis='y', length=0, pad=2)
        pos = ax_left.get_position()
        ax_left.set_position([pos.x0 - 0.01, pos.y0, pos.width, pos.height])

    # colorbar to the right
    cax = fig.add_subplot(gs[0, depth+1])
    fig.colorbar(im, cax=cax, label=cbar_label)
    return fig




def draw_sign_boundary(ax, mat: np.ndarray, thresh: float = 0, color: str = 'green', linewidth: float = 2.0) -> None:
    """
    Draw a crisp boundary along cell edges separating positive and negative
    values in a 2D matrix, aligned exactly with the grid (no interpolation).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to draw.
    mat : np.ndarray
        2D array of values (e.g., decision values).
    color : str, default 'green'
        Color of the contour line.
    linewidth : float, default 2.0
        Line width of the contour.
    """
    pos = mat > thresh
    rows, cols = mat.shape

    # Horizontal edges: sign changes between vertical neighbors
    h_edges = pos[1:, :] ^ pos[:-1, :]
    # Vertical edges: sign changes between horizontal neighbors
    v_edges = pos[:, 1:] ^ pos[:, :-1]

    segs = []

    # Vertical boundary segments (between adjacent columns)
    iy, ix = np.where(v_edges)
    for i, j in zip(iy, ix):
        x = j + 0.5
        segs.append([(x, i - 0.5), (x, i + 0.5)])

    # Horizontal boundary segments (between adjacent rows)
    iy, ix = np.where(h_edges)
    for i, j in zip(iy, ix):
        y = i + 0.5
        segs.append([(j - 0.5, y), (j + 0.5, y)])

    # Add line collection to axis
    lc = LineCollection(segs, colors=color, linewidths=linewidth, capstyle='butt')
    ax.add_collection(lc)

    # Keep axis limits aligned with pixel edges
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
