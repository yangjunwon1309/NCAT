# NCAT/__init__.py
from .NCAT_model import NCAT
from .NCAT_datageneration import NCAT_datageneration as DataGenerator
from .NN_model import DNN_LM as NNModel

__version__ = "0.1.0"

__all__ = ["NCAT", "DataGenerator", "NNModel"]