"""
Various utility functions.
"""
import numpy as np
from itertools import permutations
from fluffyclone.sampling import random_tree

def tree_list(n):
    """
    Return the list of all possible lineage trees given m mutations,
    that is, a number of (m+1)**(m-1) trees with root 0 in tuple form.
    """
    if n == 0:
        return [((),)]
    def trees(nodes):
        if len(nodes) == 1:
            return [[[nodes[0]]] + [[]]*n]
        if len(nodes) >= 2:
            treelist, current, child = [], nodes[:-1], nodes[-1]
            for oldtree in trees(current):
                for parent in [0] + current:
                    newtree = [oldtree[i].copy() for i in range(n+1)]
                    newtree[parent].append(child)
                    treelist.append(newtree)
            return treelist
    treeset = set()
    for nodes in permutations(range(1,n+1)):
        for tree in trees(list(nodes)):
            for i in range(n+1):
                tree[i].sort()
                tree[i] = tuple(tree[i])
            treeset.add(tuple(tree))
    tree0 = (tuple(range(1,n+1)),) + ((),)*n
    treelist = [tree0] + list(treeset - {tree0})
    return treelist

def distribution_estim(w, root=0, n_samples=1000,
    all=False, jit=False, empirical=False):
    """
    Empirical distribution using `n_samples` random trees.
    The result is a dictionary {tree: probability} where each tree is
    given in tuple form.
    """
    count = {}
    if all:
        from .utils import tree_list
        for tree in tree_list(w[1:,1:].shape[0]):
            count[tree] = 0
    for i in range(n_samples):
        tree = random_tree(w, root, jit)
        count[tree] = count.get(tree, 0) + 1
    return {tree: count[tree]/n_samples for tree in count}

def logweight(tree, w):
    """
    Compute the log-weight of a tree given parameter alpha.
    """
    # print(w)
    return np.sum([np.log(w[i,j]) for i, li in enumerate(tree) for j in li])

def lognorm(w):
    """
    Compute the natural logarithm of the normalisation constant
    for the weighted uniform distribution over trees.
    """
    L = -w[1:,1:]
    L -= np.diag(np.sum(L, axis=0))
    D = np.diag(w[0,1:])
    return np.linalg.slogdet(D + L)[1]

def distribution_exact(w):
    """
    Compute the probability of all trees given weight parameter w.
    The result is returned as a dictionary {tree: probability}
    with each tree given in tuple form.
    """
    trees = tree_list(w[1:,1:].shape[0])
    logz = lognorm(w)
    return {tree: np.exp(logweight(tree, w) - logz) for tree in trees}

def adjacency_matrix(tree):
    """
    Return the adjacency matrix of a tree given in tuple or dictionary form.
    """
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
    """
    Return the adjacency list of a tree given its adjacency matrix.
    The output is a dictionary x where x[i] is the children set of node i.
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
    """
    Return the adjacency list of a tree given its adjacency matrix.
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
