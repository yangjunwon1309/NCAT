import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MultiLabelBinarizer, OneHotEncoder, OrdinalEncoder
from category_encoders import BinaryEncoder, TargetEncoder

def auto_columns(X, category = "onehot"):
    X_return = X.copy()
    
    # int, float 열들에 대해서 새롭게 이름 지정
    integer_columns = [
        col for col in X.columns
        if X[col].apply(lambda x: isinstance(x, (int, float)) and x == int(x)).all()
    ]
    string_columns = [col for col in X.columns if X[col].apply(lambda x: isinstance(x, str)).all()]
    bool_columns = [col for col in X.columns if X[col].apply(lambda x: isinstance(x, bool)).all()]
    bool_columns = [[col, f"B_{i}"] for i, col in enumerate([b_col for b_col  in X.columns if (b_col in bool_columns)])]
    continuous_columns = [[col, f"C_{i}"] for i, col in enumerate([c_col for c_col  in X.columns if (c_col not in integer_columns + string_columns + bool_columns)])]
    integer_columns = [[col, f"D_{i}"] for i, col in enumerate(integer_columns)]
    
    # String 열들에 대해서 one-hot-encoding 열 생성 (int)
    if string_columns:
        if category == "onehot":
            X_encoded = pd.get_dummies(X, columns=string_columns, drop_first=False)
            new_columns = list(set(X_encoded.columns) - set(X.columns))
            X[new_columns] = X_encoded[new_columns].astype(int)
        elif category == "ordinal":
            ordinal_encoder = OrdinalEncoder()
            X[string_columns] = ordinal_encoder.fit_transform(X[string_columns])
            new_columns = string_columns
        elif category == "binary":
            encoder = BinaryEncoder(cols=string_columns)
            X_encoded = encoder.fit_transform(X)
            new_columns = list(set(X_encoded.columns) - set(X.columns))
            X[new_columns] = X_encoded[new_columns].astype(int)
        
        # Indicator의 분산이 큼 => 더 빈도 수 많음. 분산이 적음 ==> 더 빈도 수 적음.
        # 빈도 수 많음 => 서로 비슷한 class 존재할 확률이 높음(?), 즉, label의 corr이 높음(?)=> target encoding 이 적절.. 역으로 빈도 수 적다면 특이한 class 이므로, gain 이 클 수 있음 => one hot
        elif category == "dynamic":
            encoded_dfs = []
            new_columns = []
            # Calculate variance of class frequencies for each string column
            column_variances = {
                col: np.var(X[col].value_counts(normalize=True).values)
                for col in string_columns
            }

            # Determine the 25th percentile of the variances
            variance_threshold = np.percentile(list(column_variances.values()), 25)
            
            cont_column_variances = {
                col[0]: np.var(X[col[0]].values)
                for col in continuous_columns
            }
            max_cont_col = max(cont_column_variances, key=cont_column_variances.get)
            
            for col in string_columns:
                # Check if the column's variance is below or above the threshold
                if column_variances[col] <= variance_threshold:
                    # Target encode for low variance columns
                    target_encoder = TargetEncoder(smoothing=2.0)
                    encoded_col = target_encoder.fit_transform(X[[col]], X[max_cont_col])
                    encoded_col.columns = [f"{col}_target"]
                    continuous_columns = continuous_columns + [[col, f"C_{i+ len(continuous_columns)}"] for i, col in enumerate(list(set(encoded_col.columns)))]
                    encoded_dfs.append(encoded_col)
                else:
                    # One-hot encode for high variance columns
                    one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                    encoded_array = one_hot_encoder.fit_transform(X[[col]])
                    encoded_col = pd.DataFrame(
                        encoded_array,
                        columns=one_hot_encoder.get_feature_names_out([col]),
                        index=X.index
                    )
                    new_columns = new_columns + list(set(encoded_col.columns))
                    encoded_dfs.append(encoded_col)
                    
            #X = X.drop(columns=[col])
            X = pd.concat([X] + encoded_dfs, axis=1)
            
    else:
        new_columns = []
    
    #print(continuous_columns)
    #display(X)
    
    if bool_columns:
        X_bool = X[[b_cols[0] for b_cols in bool_columns]].astype(int)
        X[[b_cols[1] for b_cols in bool_columns]] = X_bool
    
    # Assign new names for one-hot-encoded columns
    one_hot_columns = [[col, f"O_{i}"] for i, col in enumerate(new_columns)]
    for original_col, new_col in one_hot_columns:
        X_return[new_col] = X[original_col]

    # Assign new names for integer and continuous columns
    for int_col in integer_columns:
        X_return[int_col[1]] = X[int_col[0]]
    for cont_col in continuous_columns:
        X_return[cont_col[1]] = X[cont_col[0]]

    # Return only the renamed columns
    return X_return[
        [col[1] for col in continuous_columns] +
        [col[1] for col in integer_columns] +
        [col[1] for col in bool_columns] +
        [col[1] for col in one_hot_columns]
    ]
