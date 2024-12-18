import unittest
import numpy as np
from NCAT import NCAT 
from NCAT.utils.auto_columns import auto_columns
import pandas as pd

class TestNCAT(unittest.TestCase):
    def test_initialization(self):
        np.random.seed(42)
        n = 200
        data = pd.DataFrame({
            "Feature1": np.random.randint(1, 101, n),
            "Feature2": np.random.randint(1, 101, n),
            "Target": np.random.choice([0, 1], n)
        })

        ncat_model = NCAT(X=auto_columns(data[["Feature1", "Feature2"]]), y=data["Target"], cluster_k=3,
                          obj="classfication", max_iter=3000, N_bins=5)

        self.assertIsNotNone(ncat_model)
        self.assertEqual(ncat_model.obj, "classfication")

if __name__ == "__main__":
    unittest.main()
