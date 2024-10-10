# Clonal trees

:::{warning}
This documentation is still a draft.
:::

## Define clonal trees

```python
from fluffyclone import ClonalTree
```

### Parent sequence encoding

```python
# Option 1: pass the parent sequence directly
tree = ClonalTree(0, 0, 0, 2, 5)

# Option 2: define parent sequence first
p = (0, 0, 0, 2, 5)
tree = ClonalTree(p)

# Option 3: add root explicitly (node 0 with no parent)
p = (None, 0, 0, 0, 2, 5)
tree = ClonalTree(p)
```
