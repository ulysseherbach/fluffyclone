"""Probabilistic tree model"""
import numpy as np

class WeightedUniform:
    """Base class for handling the weighted uniform distribution over rooted
    spanning trees on {0,1,...,n} with fixed root 0.
    Here rooted trees are seen as particular directed graphs, sometimes called
    arborescences, in which all edges point away from the root.
    """
    def __init__(self, w: np.ndarray):
        # Store edge weights as a masked array
        self._w = _w_check(w)

    @property
    def w(self):
        """Matrix of edge weights."""
        return self._w

    @property
    def size(self):
        return self._w.shape[0]

    @property
    def n_trees(self):
        n = self.size
        return (n+1)**(n-1)

    @property
    def log_partition(self):
        pass

    @property
    def log_w(self):
        pass



class TreeModel(WeightedUniform):
    """Probabilistic tree model based on the weighted uniform distribution.
    NB: d can be non-symmetric (i.e. divergence)
    """
    def __init__(self, d: np.ndarray, beta: float = 1.0):
        self._d = _d_check(d)
        self._beta = _beta_check(beta)

        # TODO : compute weight matrix
        self._compute_weights()

        # Get parent attributes
        super(TreeModel, self).__init__(w)

    def _compute_weights(self):
        """Compute the matrix of edge weights given model parameters."""
        print('Yes')
        pass

    # NB: we redfine w here to avoid direct modification
    def w(self) -> np.ndarray:
        """Return the matrix of edge weights, given model parameters."""
        return self._w.filled()

    def normalize(self):
        pass
        # Recompute d and beta

    @property
    def epsilon(self) -> float:
        return np.exp(-self._beta)




# Utility functions

def _w_check(w: np.ndarray) -> np.ma.MaskedArray:
    """Check weight matrix and return a masked version."""
    if not isinstance(w, np.ndarray):
        raise ValueError("w must be a numpy array")
    shape = w.shape
    if not len(shape) == 2:
        raise ValueError("w must be a 2D-array (square matrix)")
    if not ((shape[0] == shape[1]) and np.all(w >= 0)):
        raise ValueError("w must be a square matrix with nonnegative entries")
    # Build mask (irrelevant edges i -> 0 and i -> i)
    mask = np.zeros(shape, dtype=bool)
    mask[:, 0] = True
    mask += np.identity(shape[0], dtype=bool)
    # Build weight matrix
    w = w.astype(float, copy=True, subok=False)
    w[mask] = 0
    w = np.ma.MaskedArray(w, mask=mask, hard_mask=True, fill_value=0.0)
    return w

def _d_check(d: np.ndarray) -> np.ma.MaskedArray:
    pass


def _beta_check(beta: float) -> float:
    pass


# Tests
if __name__ == "__main__":
    w = np.ones((3,3))
    weight_matrix = _w_check(w)
    # print(weight_matrix)

    model = WeightedUniform(w)
    model.w[1,1] = 8
    print(model.w)


    test = TreeModel(w)
    # test.w()[:] = 6
    print(test.w())


