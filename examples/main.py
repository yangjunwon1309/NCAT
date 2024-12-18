
# NCAT 만들기, 서버에 올리기
# onlie tuning 지원하기
import os
import pandas as pd
import numpy as np
import sys

print("library complete")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if __name__ == "__main__":
    from NCAT import NCAT 
    from NCAT.utils.auto_columns import auto_columns

    iris = pd.read_csv(os.path.join(os.path.dirname(__file__), "Iris.csv")).drop(columns=["Id"])
    iris = auto_columns(iris, category="ordinal")

    iris_X = iris.drop(columns=["O_0"])
    iris_y = iris["O_0"]

    ncat_model = NCAT(X=iris_X, y=iris_y, cluster_k=5,
                      obj="classfication", max_iter=3000, N_bins=5)

    ncat_model.distillation(lr=0.01)

    matric, predictions = ncat_model.testing()
    print(matric)
    print("NCAT succefully generated")
    