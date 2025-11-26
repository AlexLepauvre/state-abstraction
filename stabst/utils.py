import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from typing import Optional, Sequence, Tuple
import itertools
from tqdm import tqdm
from ot import emd2
from joblib import Parallel, delayed
import itertools
import numpy as np


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


def abstract2ground_value(
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


import numpy as np

import numpy as np
from collections import defaultdict


def find_duplicate_states_across_classes(state_classes):
    """
    state_classes: list[list[tuple]]
    returns dict[state -> list[class_indices]] for states that appear in >1 class
    """
    locations = defaultdict(list)  # state -> list of class indices

    for cls_idx, cls in enumerate(state_classes):
        for s in cls:
            locations[s].append(cls_idx)

    # keep only states that appear in more than one *distinct* class
    duplicates = {s: sorted(set(idxs)) for s, idxs in locations.items()
                  if len(set(idxs)) > 1}
    return duplicates


import numpy as np

def cluster_structure(states, dim_names, coverage_tol=0.95):
    """
    states: list of tuples (one cluster)
    dim_names: list/tuple of dimension names, len = n_dims

    Prints:
      - per-dimension domain sizes
      - overall grid coverage in full space
      - for each dim: coverage of the grid when that dim is ignored
        -> tells you which dims drive the constraints.
    """
    states = np.array(states)
    if states.size == 0:
        return "(empty cluster)"

    n_dims = states.shape[1]
    uniq = np.unique(states, axis=0)

    # domains and sizes
    domains = [sorted(set(uniq[:, i])) for i in range(n_dims)]
    dom_sizes = [len(d) for d in domains]

    # full grid size and coverage
    grid_all = 1
    for s in dom_sizes:
        grid_all *= s
    n_uniq = uniq.shape[0]
    cov_all = n_uniq / grid_all if grid_all > 0 else 1.0

    lines = []
    lines.append("Grid/domain structure:")

    # per-dim domain sizes
    for i, name in enumerate(dim_names):
        vals = domains[i]
        if len(vals) <= 10:
            val_str = "{" + ",".join(map(str, vals)) + "}"
        else:
            val_str = f"{vals[0]} … {vals[-1]} (|{len(vals)}|)"
        lines.append(
            f"  {name:12s}: |domain|={dom_sizes[i]:2d}  values={val_str}"
        )

    lines.append(
        f"\n  Full grid size (product of domains): {grid_all} "
        f"  unique states: {n_uniq}  coverage={cov_all:.3f}"
    )

    # coverage when ignoring each single dimension
    lines.append("\n  Coverage when ignoring one dimension:")
    cov_except = []
    for i in range(n_dims):
        other_dims = [j for j in range(n_dims) if j != i]
        proj = uniq[:, other_dims]
        n_uniq_proj = len({tuple(row) for row in proj})
        grid_proj = 1
        for j in other_dims:
            grid_proj *= dom_sizes[j]
        cov = n_uniq_proj / grid_proj if grid_proj > 0 else 1.0
        cov_except.append(cov)
        lines.append(
            f"    drop {dim_names[i]:12s} -> grid={grid_proj:4d}, "
            f"unique={n_uniq_proj:4d}, coverage={cov:.3f}"
        )

    # highlight "constraint-driving" dims:
    # those whose removal gives very high coverage
    max_cov = max(cov_except) if cov_except else 0
    drivers = [
        dim_names[i]
        for i, cov in enumerate(cov_except)
        if cov >= coverage_tol * max_cov and cov < 1.0 - 1e-9
    ]
    if max_cov < 1.0 - 1e-9:
        # there is no dimension whose removal makes it a full grid
        lines.append("\n  No single dimension explains all missing combinations.")
    else:
        lines.append(
            "\n  Dimensions whose removal makes the remaining space "
            "almost a full grid:"
        )
        for i, cov in enumerate(cov_except):
            if abs(cov - 1.0) < 1e-6:
                lines.append(f"    {dim_names[i]}")

    return "\n".join(lines)

def _rectangular_cover(states):
    """
    states: iterable of hashable tuples (one class)
    Returns: list of rectangles, each rectangle is a list of sets:
             rect[d] = set of values allowed in dimension d.
             Each rectangle is full grid over its sets w.r.t. the given states.
    """
    # make sure states are tuples and unique
    uniq_states = {tuple(s) for s in states}
    if not uniq_states:
        return []

    D = len(next(iter(uniq_states)))
    remaining = set(uniq_states)

    # possible values in this class per dimension
    class_vals = [
        sorted({s[d] for s in uniq_states})
        for d in range(D)
    ]

    rectangles = []

    while remaining:
        seed = next(iter(remaining))
        pattern_sets = [ {seed[d]} for d in range(D) ]

        def matching(p_sets):
            """States in 'remaining' that match the pattern."""
            return {
                s for s in remaining
                if all(s[d] in p_sets[d] for d in range(D))
            }

        covered = matching(pattern_sets)

        improved = True
        while improved:
            improved = False
            best_sets = None
            best_cov = covered
            best_size = len(covered)

            # try to add one value in one dimension at a time
            for d in range(D):
                for v in class_vals[d]:
                    if v in pattern_sets[d]:
                        continue
                    candidate_sets = [set(x) for x in pattern_sets]
                    candidate_sets[d].add(v)

                    cand_states = matching(candidate_sets)

                    # full-grid check inside this rectangle w.r.t. remaining
                    prod = 1
                    for s in candidate_sets:
                        prod *= len(s)

                    if len(cand_states) == prod and len(cand_states) > best_size:
                        best_size = len(cand_states)
                        best_cov = cand_states
                        best_sets = candidate_sets

            if best_sets is not None:
                pattern_sets = best_sets
                covered = best_cov
                improved = True

        rectangles.append(pattern_sets)
        remaining -= covered

    return rectangles


def compact_class_repr(states, dim_names):
    """
    states: list of tuples for one class
    dim_names: sequence of names, length = #dims
    Returns a string like:
        class ≈ {[0,1,2], [1,2], 1, [3], [1,2]} ∪ {...}
    """
    rects = _rectangular_cover(states)
    parts = []
    for rect in rects:
        dims_repr = []
        for i, vals in enumerate(rect):
            vals = sorted(vals)
            if len(vals) == 1:
                dims_repr.append(f"{dim_names[i]}: {str(vals[0])}")
            else:
                dims_repr.append(f"{dim_names[i]}: [{",".join(map(str, vals))}]")
        parts.append("{" + ", ".join(dims_repr) + "}")
    return "\n ∪ \n".join(parts)



def summarize_state_classes(
    state_classes,
    eps,
    dim_names=("energy", "offer", "current_cost", "future_cost", "time"),
    max_classes=10,
    bar_width=20,
    tight_threshold=0.1,   # relative range <= 10% of global -> "tight"
    loose_threshold=0.5    # relative range >= 50% of global -> "spread"
):
    """
    state_classes: list of classes, each class = list of states
                   each state = tuple (energy, offer, current_cost, future_cost, time)
    """

    # --- basic sanity ---
    non_empty_classes = [c for c in state_classes if len(c) > 0]
    num_classes = len(non_empty_classes)
    if num_classes == 0:
        print("No non-empty classes.")
        return
    if num_classes < 50:
        max_classes = num_classes
    sizes = np.array([len(c) for c in non_empty_classes])
    total_states = int(sizes.sum())

    print("=" * 80)
    print(f"AGGREGATION SUMMARY AT EPS={eps}")
    print("=" * 80)
    print(f"# classes                : {num_classes}")
    print(f"# states total           : {total_states}")
    print(f"class size - mean / std  : {sizes.mean():.2f} / {sizes.std():.2f}")
    print(f"class size - min / max   : {sizes.min()} / {sizes.max()}")
    print()

    # --- global state-space stats ---
    all_states = np.array(
        [s for cls in non_empty_classes for s in cls],
        dtype=float
    )
    global_min = all_states.min(axis=0)
    global_max = all_states.max(axis=0)
    global_range = global_max - global_min

    print("GLOBAL STATE-SPACE STATS (over all states):")
    print("-" * 80)
    for i, name in enumerate(dim_names):
        r = global_range[i]
        print(
            f"{name:>12}: min={global_min[i]:8.3f}  "
            f"max={global_max[i]:8.3f}  range={r:8.3f}"
        )
    print()

    # --- average within-class range per dimension ---
    cls_ranges = []
    for cls in non_empty_classes:
        if len(cls) <= 1:
            continue
        arr = np.array(cls, dtype=float)
        cls_ranges.append(arr.max(axis=0) - arr.min(axis=0))

    if cls_ranges:
        cls_ranges = np.vstack(cls_ranges)
        mean_within = cls_ranges.mean(axis=0)
        rel_within = np.divide(
            mean_within,
            global_range,
            out=np.zeros_like(mean_within),
            where=global_range != 0,
        )

        print("AVERAGE WITHIN-CLASS RANGE PER DIMENSION:")
        print("(relative = within-class range / global range)")
        print("-" * 80)
        for i, name in enumerate(dim_names):
            print(
                f"{name:>12}: mean_range={mean_within[i]:8.3f}  "
                f"relative={rel_within[i]:6.3f}"
            )
        print()

    # --- helper for ASCII bar ---
    def bar(rel):
        length = int(round(rel * bar_width))
        length = max(0, min(bar_width, length))
        return "#" * length + "." * (bar_width - length)

    # --- show per-class summaries (largest classes first) ---
    sorted_idx = np.argsort(-sizes)  # minus for descending

    print("PER-CLASS SUMMARIES (largest classes first):")
    print("(relative_range = class_range / global_range)")
    print("-" * 80)

    for rank, idx in enumerate(sorted_idx[:max_classes], start=1):
        cls = non_empty_classes[idx]
        arr = np.array(cls, dtype=float)
        n = arr.shape[0]

        cls_min = arr.min(axis=0)
        cls_max = arr.max(axis=0)
        vals = [np.unique(arr[:, i]) for i in range(len(dim_names))]
        cls_range = cls_max - cls_min
        cls_rel_range = np.divide(
            cls_range,
            global_range,
            out=np.zeros_like(cls_range),
            where=global_range != 0,
        )

        # classify dimensions as tight/loose
        tight_dims = [dim_names[i] for i, v in enumerate(cls_rel_range)
                      if v <= tight_threshold]
        loose_dims = [dim_names[i] for i, v in enumerate(cls_rel_range)
                      if v >= loose_threshold]

        print(f"Class #{rank} (original index {idx})")
        print(f"  #states: {n}")
        if tight_dims:
            print(f"  Tight dims (≈constant): {', '.join(tight_dims)}")
        if loose_dims:
            print(f"  Spread dims (high var): {', '.join(loose_dims)}")

        print("Compact representation:")
        print("  class ≈")
        print(compact_class_repr(cls, dim_names))
        print("-" * 80)

