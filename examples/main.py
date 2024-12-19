
# NCAT 만들기, 서버에 올리기
# onlie tuning 지원하기
import os
import pandas as pd
import numpy as np
import sys
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, accuracy_score, f1_score
# print("library complete")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if __name__ == "__main__":
    from NCAT import NCAT 
    from NCAT.utils.auto_columns import auto_columns

    iris = pd.read_csv(os.path.join(os.path.dirname(__file__), "Iris.csv")).drop(columns=["Id"])
    iris = auto_columns(iris, category="ordinal")

    iris_X = iris.drop(columns=["O_0"])
    iris_y = iris["O_0"]

    ncat_model = NCAT(X=iris_X, y=iris_y, cluster_k=5,
                      obj="classfication", max_iter=5000, min_iter=1000, N_bins=5)

    # ncat_model.distillation(lr=0.01)

    ncat_model.distillation_multi(lr = 0.07)

    matric, predictions = ncat_model.testing()
    CAT_pred = pd.Series(ncat_model.ncat.catboost.predict(ncat_model.X_test).ravel())

    print(matric)
    print(r2_score(ncat_model.y_test, predictions), ": r2 of NN",
          np.sqrt(mean_squared_error(ncat_model.y_test, predictions)), ": rmse of NN")
    print(r2_score(ncat_model.y_test, CAT_pred), ": r2 of CAT",
          np.sqrt(mean_squared_error(ncat_model.y_test, CAT_pred)), ": rmse of CAT")

    ncat_model.save(file_name="iris_0")

    print("NCAT succefully generated")
    