import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Tuple
import itertools

import numpy as np


def bisim_metric_fast(tp, rewards, gamma=0.9, tol=1e-6, max_iters=10000):
    """
    Fast Ferns-style bisimulation metric approximation.
    Avoids Kantorovich/OT; uses expected next-state distances directly.

    tp : np.ndarray, shape (nS, nA, nS)  -- transition probabilities
    rewards : np.ndarray, shape (nS, nA)
    gamma : float in [0,1)
    """
    nS, nA = rewards.shape
    d = np.zeros((nS, nS))

    for _ in range(max_iters):
        d_old = d.copy()

        for i in range(nS):
            for j in range(i + 1, nS):
                vals = []
                for a in range(nA):
                    r_diff = abs(rewards[i, a] - rewards[j, a])
                    # expected next-state distance difference
                    exp_diff = np.sum(np.abs(tp[i, a, :] - tp[j, a, :]) @ d)
                    vals.append(r_diff + gamma * exp_diff)
                d[i, j] = d[j, i] = max(vals)
        np.fill_diagonal(d, 0.0)

        if np.max(np.abs(d - d_old)) < tol:
            break
    return d


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
               for state, idx in enumerate(list(itertools.chain(*classes)))}
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
            ax.axhline(b, color='white', lw=0.1 + 0.25*(depth - i), alpha=0.5)
            ax.axvline(b, color='white', lw=0.1 + 0.25*(depth - i), alpha=0.5)

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
