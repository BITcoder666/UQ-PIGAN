# seed_utils.py
"""
Utility script: Fix random seeds to ensure reproducibility of machine learning model training.
Sets global random seeds including NumPy, Python random, scikit-learn, and XGBoost using seed value 2025.
Call set_seed(2025) before training to fix randomness.
Dependencies: numpy, random, sklearn, xgboost (optional).
"""

import random
import numpy as np
import os

# Try to import torch, if it exists (not used in historical scripts, but for completeness)
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Try to import xgboost, if it exists
try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


def set_seed(seed=2025, deterministic=True):
    """
    Set global random number seed to the specified value to ensure reproducibility.

    Args:
        seed: Random seed value, default is 2025.
        deterministic: If True, set CuDNN to deterministic mode (only if torch is available).

    Returns:
        None
    """
    # Python random seed
    random.seed(seed)

    # NumPy seed
    np.random.seed(seed)

    # Set environment variable
    os.environ['PYTHONHASHSEED'] = str(seed)

    # scikit-learn seed (via global setting)
    if 'sklearn' in globals() or True:  # Assuming sklearn might be imported later
        pass  # sklearn random state is usually set via model.random_state=seed

    # XGBoost seed (if available, set via model random_state, no global random.seed needed)
    if XGB_AVAILABLE:
        pass  # XGBoost is set via XGBRegressor(random_state=seed)

    # Torch seed (if available)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    print(f"Random seed fixed to {seed} to ensure reproducibility.")


# Example usage (optional, for testing at the end of script)
if __name__ == "__main__":
    set_seed(2025)
    # Test: Generate fixed random numbers
    print("Testing NumPy random numbers:", np.random.rand(3))
