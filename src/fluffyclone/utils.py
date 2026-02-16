"""Various utility functions."""
import numpy as np
from itertools import permutations
from fluffyclone.sampling import random_tree

def tree_list(n):
    """List of all possible lineage trees given m mutations.

    That is, a number of (n+1)**(n-1) trees with root 0 in tuple form.
    """
    if n == 0:
        return [((),)]
    def trees(nodes):
        if len(nodes) == 0:
            return None
        if len(nodes) == 1:
            return [[[nodes[0]]] + [[]]*n]
        treelist, current, child = [], nodes[:-1], nodes[-1]
        for oldtree in trees(current):
            for parent in [0, *current]:
                newtree = [oldtree[i].copy() for i in range(n+1)]
                newtree[parent].append(child)
                treelist.append(newtree)
        return treelist
        # return None
    treeset = set()
    for nodes in permutations(range(1,n+1)):
        for tree in trees(list(nodes)):
            for i in range(n+1):
                tree[i].sort()
                tree[i] = tuple(tree[i])
            treeset.add(tuple(tree))
    tree0 = (tuple(range(1,n+1)),) + ((),)*n
    return [tree0, *list(treeset - {tree0})]

def distribution_estim(w, root=0, n_samples=1000,
    all=False, jit=False):
    """Empirical distribution using `n_samples` random trees.

    The result is a dictionary {tree: probability} where each tree is
    given in tuple form.
    """
    count = {}
    if all:
        for tree in tree_list(w[1:,1:].shape[0]):
            count[tree] = 0
    for i in range(n_samples):
        tree = random_tree(w, root, jit)
        count[tree] = count.get(tree, 0) + 1
    return {tree: count[tree]/n_samples for tree in count}

def log_weight(tree, a):
    """Compute the log-weight of a tree given log-weight parameters."""
    return np.sum([a[i,j] for i, li in enumerate(tree) for j in li])

def log_weight_total(w):
    """Natural logarithm of the normalisation constant.

    That is, the logarithm of the total weight of all spanning trees
    of a weighted digraph that are rooted in node 0 given its weight
    matrix. Returns -inf if the total weight is 0.
    """
    l = -w[1:,1:]
    l -= np.diag(np.sum(l, axis=0))
    d = np.diag(w[0,1:])
    return np.linalg.slogdet(d + l)[1]

def distribution_exact(w):
    """Probability of all trees given weight parameter w.

    The result is returned as a dictionary {tree: probability}
    with each tree given in tuple form.
    """
    trees = tree_list(w[1:,1:].shape[0])
    logz = log_weight_total(w)
    a = np.log(w*(w > 0) + 1*(w == 0))
    a[w == 0] = -np.inf
    return {tree: np.exp(log_weight(tree, a) - logz) for tree in trees}

def norm_weight_matrix(w):
    """Normalize the weight matrix to ensure identifiability.

    NB: The output corresponds to the forest parameterization.
    """
    n = np.size(w[0, 1:])
    w_norm = np.zeros((n+1, n+1))
    for j in range(1, n+1):
        w_norm[:, j] = w[:, j] / w[0, j]
    return w_norm

def forest_matrix(w):
    """Matrix of out-forests (see Chebotarev2002)."""
    n = np.size(w[0, 1:])
    w_norm = norm_weight_matrix(w)
    # Compute Laplacian matrix
    laplacian = -w_norm[1:, 1:]
    laplacian -= np.diag(np.sum(laplacian, axis=0))
    return np.linalg.inv(np.eye(n) + laplacian)

def edge_probability_matrix(w):
    """Matrix of direct edge probabilities (exact)."""
    n = np.size(w[0, 1:])
    q = forest_matrix(w)
    p = np.zeros((n+1, n+1))
    p[0, 1:] = np.diag(q)
    for i in range(n):
        for j in range(n):
            p[i+1, j+1] = (q[j, j] - q[j, i]) * w[i+1, j+1] / w[0, j+1]
    return p

def filiation_probability_matrix(w):
    """Matrix of filiation probabilities (exact, 'fast' version)."""
    n = np.size(w[0, 1:])
    f = np.zeros((n+1, n+1))
    # Total weight of all trees
    lz = log_weight_total(w)
    for i in range(n+1):
        for j in range(n+1):
            # Germline and self paths
            if i in (0, j):
                f[i, j] = 1
            elif j > 0:
                # Non-trivial paths
                # 1. Build state vector u of subtrees rooted in node i
                # within n mutations and containing node j. That is, u
                # has length 2**(n-2) and is defined in lexicographic
                # order by u[s] = (x[s,0], x[s,1], ..., x[s,n]) with
                # x[s,k] = 1 if node k belongs to the subtree, and 0
                # otherwise.
                # NB: Every u[s] hence satisfies x[s,i] = x[s,j] = 1
                # and x[s,0] = 0.
                fixed = {0: [0], i: [1], j: [1]}
                val = {k: fixed.get(k, [0, 1]) for k in range(n+1)}
                u = [(k,) for k in val[n]]
                for node in range(n-1, -1, -1):
                    u = [(k, *v) for k in val[node] for v in u]
                # 2. Integrate the probability over the 2**(n-2) configs
                b = w.copy()
                b[i] = 0
                p = 0
                for state in u:
                    s = np.array(state)
                    x1 = np.array(np.nonzero(s))
                    # Key step:
                    # rearrange elements so that i always appears first
                    r = np.cumsum(s) - 1
                    x1[0, [0, r[i]]] = x1[0, [r[i], 0]]
                    s[i] = 0
                    x2 = np.array(np.nonzero(1-s))
                    # print(x1, x2)  # Show configurations
                    # Once everything is sorted out do the computation
                    l1 = log_weight_total(w[np.transpose(x1), x1])
                    l2 = log_weight_total(b[np.transpose(x2), x2])
                    p += np.exp(l1 + l2 - lz)
                # Store the probability
                f[i, j] = p
    return f

def adjacency_matrix(tree):
    """Adjacency matrix of a tree given in tuple or dictionary form."""
    if isinstance(tree, dict):
        m = max([max(children) for children in tree.values()]) + 1
        z = np.zeros((m,m), dtype=np.uint8)
        for i, children in tree.items():
            for j in children:
                z[i,j] = 1
    if isinstance(tree, tuple):
        m = len(tree)
        z = np.zeros((m,m), dtype=np.uint8)
        for i in range(m):
            for j in tree[i]:
                z[i,j] = 1
    return z

def dict_from_adjacency(z):
    """Return the adjacency list of a tree given its adjacency matrix.

    The output is a dictionary {i: children[i]}.
    """
    m = z[0].size
    tree = {}
    for i in range(m):
        children = []
        for j in range(m):
            if z[i,j] == 1:
                children.append(j)
        if len(children) > 0:
            tree[i] = set(children)
    return tree

def tuple_from_adjacency(z):
    """Adjacency list of a tree given its adjacency matrix.

    The output is a tuple x where x[i] is the children list of node i,
    itself in the form of a tuple arranged in ascending order.
    """
    m = z[0].size
    tree = []
    for i in range(m):
        children = []
        for j in range(m):
            if z[i,j] == 1:
                children.append(j)
        tree.append(tuple(children))
    return tuple(tree)


# Tests
if __name__ == '__main__':

    # Tuple form
    tree = ((3,), (), (), (1, 2))
    print(adjacency_matrix(tree))

    # Dictionary form
    tree = {0: {3}, 3: {1, 2}}
    print(adjacency_matrix(tree))

    # Recover dictionaary form
    print(dict_from_adjacency(adjacency_matrix(tree)))

    # Tree list
    print(tree_list(2))

    # Number of trees
    print(len(tree_list(2)))
    print(len(tree_list(3)))
    print(len(tree_list(4)))
    print(len(tree_list(5)))
