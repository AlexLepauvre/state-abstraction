import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Tuple
import itertools

import numpy as np


def evaluate_policy(states, tp, reward, policy, tdim=4):
    """
    Compute expected return under a given policy for a finite-horizon MDP.
    - states: list of tuples including time
    - tp: (S,A,S)
    - reward: (S,A)
    - policy: 
        if deterministic -> shape (S,) with integer actions
        if stochastic -> shape (S,A) with probs summing to 1
    Returns
    -------
    V : (S,)
    Q : (S,A)
    """
    S, A, _ = tp.shape
    V = np.zeros(S)
    Q = np.zeros((S, A))
    T = sorted({s[tdim] for s in states})

    # backward sweep
    for t in reversed(T):
        idx_t = [i for i, s in enumerate(states) if s[tdim] == t]
        for i in idx_t:
            for a in range(A):
                Q[i,a] = reward[i,a] + tp[i,a,:] @ V
            # deterministic policy
            if policy.ndim == 1:
                V[i] = Q[i, policy[i]]
            else:
                V[i] = (policy[i] * Q[i]).sum()
    return V, Q


def backward_induction(
    states,
    transition_proba,   # shape (S, A, S)
    reward,             # shape (S, A)
    tdim: int = 5
):
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
    n_states = len(states)
    n_actions = reward.shape[1]
    state_to_idx = {state: idx for idx, state in enumerate(states)}
    # Initialize V and Q:
    V = np.zeros(n_states)
    Q = np.zeros((n_states, n_actions))
    # Extract time points:
    T = sorted(list(set([state[tdim] for state in states])))

    # Loop through time steps:
    for t in reversed(T):
        for a in range(n_actions):
            for state in states:
                if state[tdim] != t:
                    continue
                Q[state_to_idx[state], a] = reward[state_to_idx[state], a] + np.dot(transition_proba[state_to_idx[state], a, :], V)
                V[state_to_idx[state]] = np.max(Q[state_to_idx[state], :])
    return V, Q


def plot_dv(DV: np.ndarray, C, O, E, T, state_to_idx):
    """
    Plot decision values across offers and cost transitions.

    Parameters
    ----------
    DV : np.ndarray, optional
        Decision value array. If None, computed internally.
    """

    fig, ax = plt.subplots(len(C) * len(C) , len(O), figsize=[12, 8])
    fig.suptitle('Decision values across offers and costs transitions', size=14)
    for o_i, o in enumerate(O):
        ctr = 0
        for cc in C:
            for fc in C:
                mat = np.zeros([len(E), len(T[:-1])])
                for i, e in enumerate(E):
                    for ii, t in enumerate(T[:-1]):
                        mat[i, ii] = DV[state_to_idx[(e, o, cc, fc, t)]] 
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
                if ctr + 1 == len(C) * len(C):
                    ax[ctr, o_i].set_xlabel('Trials', size=12)
                else:
                    ax[ctr, o_i].set_xticklabels([])
                ctr += 1
    plt.tight_layout()
    cbar = fig.colorbar(im, ax=ax.ravel().tolist(),fraction=0.025, pad=0.01)
    cbar.ax.set_ylabel('Decision value', size=16)
    return fig, ax


def full_state_value(classes, V_reduced=None, Q_reduced=None, S=None):
    """

    classes: list[list[int]] partition of states.
    V_reduced: (K,), Q_reduced: (K, A). Either may be None.
    S: optional total #states. If None, inferred from classes.
    Returns: (V_full (S,), Q_full (S, A), class_of_state (S,))
    """
    if S is None:
        S = max(i for cls in classes for i in cls) + 1

    class_of_state = np.full(S, -1, dtype=int)
    for k, idxs in enumerate(classes):
        class_of_state[np.asarray(idxs, dtype=int)] = k

    if (class_of_state == -1).any():
        missing = np.flatnonzero(class_of_state == -1)[:10]
        raise ValueError(f"Some states not in any class, e.g. {missing}…")

    V_full = None if V_reduced is None else np.asarray(V_reduced)[class_of_state]
    Q_full = None if Q_reduced is None else np.asarray(Q_reduced)[class_of_state, :]

    return V_full, Q_full, class_of_state


def reduce_mdp(classes, tp, reward):
    """
    Collapse an MDP given exact bisimulation classes.
    classes: list[list[int]] where each inner list contains original state indices.
    tp: (S, A, S) transition probs
    reward: (S, A) rewards
    Returns: (tp_reduced (K, A, K), reward_reduced (K, A), class_of_state (S,))
    """
    S, A, _ = tp.shape
    K = len(classes)

    # Map each original state -> its class
    class_of_state = np.empty(S, dtype=int)
    for k, idxs in enumerate(classes):
        class_of_state[idxs] = k

    # One-hot class indicator matrix M: (S, K), used to sum probs into classes
    M = np.zeros((S, K), dtype=float)
    for k, idxs in enumerate(classes):
        M[idxs, k] = 1.0

    # Average rewards within each class
    reward_reduced = np.empty((K, A), dtype=float)
    for k, idxs in enumerate(classes):
        reward_reduced[k, :] = reward[idxs, :].mean(axis=0)

    # Average transitions within each class, then pool to classes
    tp_reduced = np.empty((K, A, K), dtype=float)
    for k, idxs in enumerate(classes):
        for a in range(A):
            # mean over members of class k → distribution over original S
            avg_row = tp[idxs, a, :].mean(axis=0)          # (S,)
            # pool S → K
            tp_reduced[k, a, :] = avg_row @ M              # (K,)

    return tp_reduced, reward_reduced, class_of_state


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
