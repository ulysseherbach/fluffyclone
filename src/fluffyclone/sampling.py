"""
Sampling trees using Wilson's method.
"""
import numpy as np

# Core functions

def random_walk_transition_matrix(w, root):
    """
    Compute the transition matrix of the random walk on the directed graph
    defined by `w` (matrix of rates whose diagonal elements must be zero).

    NB: This is an intermediate function in Wilson's method for simulating
    random spanning trees from the weighted uniform distribution. Here we
    construct in-trees so all directions are reversed with respect to `w`.

    Parameters
    ----------

    w: ndarray
        Matrix of directed edge weights, interpreted as transition rates of
        the random walk. Diagonal elements must be zero (i.e. no self-edge).

    root: int
        Underlying root for the trees (the column w[:,root] is ignored).

    Returns
    -------

    p: ndarray
        Transition matrix of the corresponding random walk (Markov chain).
    """
    # Compute the exit rate from each node
    s = np.sum(w, axis=0)
    # Compute the transition matrix
    m = w.shape[0]
    p = np.zeros((m, m))
    for i in range(m):
        if i != root:
            if s[i] > 0:
                p[i] = w[:, i] / s[i]
            else:
                raise ValueError(f"No edge towards vertex {i}.")
    return p

def random_walk_step(x, p):
    """
    Perform one step of random walk from state `x` with transition matrix `p`.
    """
    if np.abs(np.sum(p[x]) - 1) > 1e-4:
        print(p[x], np.sum(p[x]))
        raise ValueError("Sum of transition probabilities must be 1.")
    # # Option 1: basic implementation
    # return np.dot(np.arange(p[x].size), np.random.multinomial(1, p[x]))
    # Option 2: numba-friendly implementation
    return np.searchsorted(np.cumsum(p[x]), np.random.random(), side="right")

def loop_erasure(path):
    """
    Compute the loop erasure of a given path.
    """
    if path[0] == path[-1]:
        return [path[0]]
    else:
        i = np.max(np.arange(len(path))*(np.array(path)==path[0]))
    if path[i+1] == path[-1]:
        return [path[0], path[i+1]]
    else:
        return [path[0]] + loop_erasure(path[i+1:])

def core_random_tree(w, root, step):
    """
    Core function for sampling a weighted uniform spanning tree.
    """
    # Check weight matrix elements
    if np.any(np.diag(w)):
        raise ValueError("Diagonal elements must be zero.")
    if np.any(w < 0):
        raise ValueError("Edge weights must be nonnegative.")
    # Initialization
    m = w.shape[0]
    tree = [[] for i in range(m)]
    v = {root} # Vertices of the tree
    r = list(set(range(m)) - v) # Remaining vertices
    np.random.shuffle(r) # Random order
    r = list(r)
    # Compute the transition matrix
    p = random_walk_transition_matrix(w, root)
    # Main loop of Wilson's method
    while len(r) > 0:
        state = r[0]
        path = [state]
        # Compute a random path that reaches the current tree
        while path[-1] not in v:
            state = step(path[-1], p)
            path.append(state)
        path = loop_erasure(path)
        # Append the loop-erased path to the current tree
        for i in range(len(path)-1):
            v.add(path[i])
            r.remove(path[i])
            tree[path[i+1]].append(path[i])
    for i in range(m):
        tree[i].sort()
    # Convert the tree into tuple form
    tree = tuple([tuple(tree[i]) for i in range(m)])
    return tree

# Main function

def random_tree(w, root=0, jit=False):
    """
    Generate a random spanning tree from the weighted uniform distribution
    using Wilson's method, given matrix `w` of directed edge weights.

    Parameters
    ----------

    w: ndarray
        Matrix of directed edge weights (nonnegative entries).
        NB: diagonal elements must be zero (i.e. no self-edge).

    root: int
        Root of the tree (the column w[:,root] is ignored).

    jit: bool
        If true, use just-in-time compilation from the Numba package.
        NB: Only useful when jit=False is slow (compilation takes about 2s).

    Returns
    -------

    tree: tuple[tuple]
        Generated tree in tuple form (tree[i] contains children of vertex i).
    """
    if jit:
        # Just-in-time compiled version
        from .compiled import random_walk_step_jit
        step = random_walk_step_jit
    else:
        step = random_walk_step
    tree = core_random_tree(w, root, step)
    return tree


# Tests
if __name__ == "__main__":

    # Number of nodes without root
    n = 3

    # Weight matrix
    w = np.zeros((n+1,n+1))
    w[0,1] = 1
    w[1,2] = 2
    w[2,1] = 1
    w[1:3,3] = 1
    print("Weight matrix:")
    print(w)

    # Transition matrix
    p = random_walk_transition_matrix(w, root=0)
    print("Transition matrix:")
    print(p)

    # Random walk step
    x = 1
    print("Random step (in-tree):")
    y = random_walk_step(x, p)
    print(f"{x} -> {y}")

    # Sample a tree
    tree = random_tree(w)
    print(tree)
