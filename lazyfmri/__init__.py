# lazyfmri/__init__.py

# Now import core functionality (optional, but typical in __init__.py)
from . import utils
from . import plotting
from . import dataset
from . import fitting
from . import preproc

__all__ = ["utils", "plotting", "dataset", "fitting", "preproc"]
