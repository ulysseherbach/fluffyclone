# Distributions

:::{warning}
This documentation is still a draft.
:::

## Definition

```python
from fluffyclone import WeightedUniform, WeightedUniformGibbs

# Number of clones
n_clones = 5

# General form
model1 = WeightedUniform(n_clones)

# Gibbs form
model2 = WeightedUniformGibbs(n_clones)

# Using just-in-time-compilation
model3 = WeightedUniform(n_clones, jit=True)
model4 = WeightedUniformGibbs(n_clones, jit=True)
```

## Usage

```python
# Sampling trees from the model
tree = model1.sample()
tree_list = model1.sample(size=10)
```
