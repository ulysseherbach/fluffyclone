# Building the documentation

Use python `3.10` or `3.11` but not `3.12` because `sphinxcontrib.collections`
depends on something that is deprecated and removed in `3.12`.

## Install dependencies

From the root of the project you can install the dependencies required for the 
documentation generation by typing:

```bash
pip install -r docs/requirements.txt
```

## Build

To build the website run the command:

```bash
sphinx-build docs docs/_build/html
```

or go the `docs` folder and run `make html`.

## Visualize the website

You can open the `docs/_build/html/index.html` file in your browser.
