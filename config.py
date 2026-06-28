"""Shared training state: dataset arrays, model cache, and hyperparameters.

All variables here are mutated directly by run_datasets.py and
get_or_train_split_model in core.py. Reset model_cache between dataset
evaluations to avoid cross-contamination (the test suite does this via
an autouse fixture in conftest.py).
"""

model_cache = {}
X = []
y = []
X_test = []
y_test = []
C = 0.1  # LogisticRegression regularisation strength
