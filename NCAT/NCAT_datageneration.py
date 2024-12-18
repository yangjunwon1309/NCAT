import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from datetime import datetime, date

import matplotlib.pyplot as plt
import math
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MultiLabelBinarizer, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from category_encoders import BinaryEncoder, TargetEncoder

from catboost import CatBoostClassifier, CatBoostRegressor, Pool


class gibbs_generator():
    def __init__(self, j =0, X=pd.DataFrame(), iter = 3):
        self.X = X
        self.j = j
        self.iter = iter
    
    def conditional_df(self):
        conditional_list = []

        col_count = len(self.X.columns)
        X_values = self.X#.sample(n=min(len(self.X), 250), random_state=42).values

        for idx in range(len(X_values)):
            row = X_values.iloc[idx, :]
            for col_idx in range(col_count):
                row_i = row.copy()
                row_i.iloc[col_idx] = np.nan  # Replace the target column with NaN
                conditional_list.append(np.hstack([row_i, row.iloc[col_idx]]))

        conditional_array = pd.DataFrame(
            np.array(conditional_list),
            columns=[str(_) for _ in range(col_count)] + ["i"]
        )
        
        std_values = self.X.std().values
        
        def conditional_prob(row, range_multiplier=1.0):
            cond_prob = 1.0

            row_values = row[:-1].values
            target_value = row["i"]
            known_mask = ~np.isnan(row_values)

            if np.all(known_mask):
                return cond_prob

            unknown_indices = np.where(~known_mask)[0]
            known_indices = np.where(known_mask)[0]
            known_values = row_values[known_indices]

            feature_ranges = std_values[known_indices] * range_multiplier #* 1.5

            feature_array = self.X.iloc[:, known_indices].values
            mask = np.all(np.abs(feature_array - known_values) <= feature_ranges, axis=1)
            subset = self.X[mask]
            
            for index in unknown_indices:
                x_i_std = std_values[index] * range_multiplier
                prob_i = (
                    np.sum(np.abs(subset.iloc[:, index].values - target_value) <= x_i_std)
                    / len(subset)
                    if len(subset) > 0
                    else 1e-10
                )
                cond_prob *= prob_i

            return cond_prob

        conditional_array["prob"] = conditional_array.apply(
            lambda x: conditional_prob(x, range_multiplier=0.5), axis=1
        )
        conditional_array.fillna(0, inplace=True)

        self.conditional_array = conditional_array
        # print("conditional_array complete")

        return conditional_array
    
    def model_make(self):
        conditional_array = self.conditional_df()

        X_whole = conditional_array.drop(columns = ["prob"])
        y_whole = conditional_array[["prob"]]

        self.base_model = LinearRegression(fit_intercept = True)
        self.base_model.fit(X_whole, y_whole)
        # print(r2_score(y_whole, self.base_model.predict(X_whole)), mean_squared_error(y_whole, self.base_model.predict(X_whole)))

        # print(self.base_model.coef_)
        # plt.clf()
        # plt.plot([0, 1], [0, 1])
        # plt.scatter(y_whole, self.base_model.predict(X_whole))
        # plt.show()
    
    def gibbs_data(self):
        gibbs_jth_features = self.X.copy()

        def gibbs_sampling(row, change_n, model, bounds_multiplier=0.5, num_samples=100):
            data = row.values.copy()
            data[change_n] = 0
            data = np.hstack([data, np.array([0])])
            mean_val = gibbs_jth_features.iloc[:, change_n].mean()
            std_val = gibbs_jth_features.iloc[:, change_n].std()
            
            lowest = min(gibbs_jth_features.iloc[:, change_n])
            highest = max(gibbs_jth_features.iloc[:, change_n])
            
            # Bound 설정 (mean ± bounds_multiplier * std)
            lower_bound = lowest #mean_val - bounds_multiplier * std_val
            upper_bound = highest #mean_val + bounds_multiplier * std_val
            
            # linspace 생성
            candidate_values = np.linspace(lower_bound, upper_bound, num_samples)
            
            # 후보 값들로 데이터를 변형해 예측
            data_candidates = np.tile(data, (num_samples, 1))
            data_candidates[:, -1] = candidate_values
            data_candidates = pd.DataFrame(data_candidates, columns= self.conditional_array.columns[:-1])
            predictions = model.predict(data_candidates)
            
            # 예측값 기반 분포 생성
            probabilities = np.exp(np.square(predictions))
            probabilities /= probabilities.sum()
            probabilities = probabilities.flatten()
            # 분포에 따라 샘플링
            sampled_value = np.random.choice(candidate_values, p=probabilities)
            
            if "C" not in self.X.columns[change_n]:
                sampled_value = int(sampled_value + 0.5)
            return sampled_value

        iter = self.iter
        for _ in range(iter):
            for idx in range(len(gibbs_jth_features)):
                change_n = np.random.randint(0, len(gibbs_jth_features.columns))
                row = gibbs_jth_features.iloc[idx]
                sampled_value = gibbs_sampling(row, change_n, self.base_model)
                
                gibbs_jth_features.iloc[idx, change_n] = sampled_value
        return gibbs_jth_features


class NCAT_datageneration():
    
    def __init__(self, X_train, y_train, iter = 3000, depth = 2, lr=0.01, k =7, obj = "regression", N_bins = 10):
        self.X = X_train #auto_columns(X_train) # X_train
        self.y = y_train
        if obj == "classfication":
            self.y_class = len(self.y.unique())
        self.iter = iter
        self.N_sub = int(self.iter*0.4)
        self.depth = depth
        self.lr = lr
        self.max_fn = int(np.sqrt(self.X.shape[1]))
        self.N_bins = N_bins
        self.k = k
        self.bounds = self.boundary(self.N_bins)
        self.obj = obj
        
        self.make_catboost_dir()
        self.Generated_Data = pd.DataFrame()
    
    
    ## 여기서 Classify 면 Classify 모델 만들어서 predict 하도록,
    ## Reg면 Reg로 predict 하도록..
    
    def make_catboost_dir(self):
        try:
            catboost_dir = os.path.join(os.getcwd().replace("\\", "/"), f'NCAT_catboost_{self.obj}')
            if not os.path.exists(catboost_dir):
                os.makedirs(catboost_dir)
        except Exception as e:
            print(e)
            print("setitng catboost directory as User download")
            catboost_dir = os.path.join(os.path.expanduser("~"), "Downloads")
            
        self.catboost_dir = catboost_dir
    
    def boundary(self, n_bins = 10):
        continuous_columns = [c_col for c_col in self.X.columns if "C" in c_col]
        integer_columns = [c_col for c_col in self.X.columns if ("D" in c_col) or ("O" in c_col) or ("B" in c_col)]
        
        conti_bounds = np.array([self.X[continuous_columns].min(axis=0).values, self.X[continuous_columns].max(axis=0).values])
        int_bounds = np.array([self.X[integer_columns].min(axis=0).values, self.X[integer_columns].max(axis=0).values])
        
        lower_bounds = np.hstack((conti_bounds[0], int_bounds[0]))
        upper_bounds = np.hstack((conti_bounds[1], int_bounds[1]))

        # 최종 bounds
        bounds = np.array([lower_bounds, upper_bounds])
        return bounds
    
    def y_bin(self):
        n_bins = self.N_bins
        return pd.cut(self.y, bins=n_bins, labels=False, retbins=True)[1]
    
    # RF 생성 및 학습 (sklearn)
    def Grid_Search_CATBoost(self):
        # grid 지정
        # pool 지정
        # valid 에 대해서 검증, 가장 나은 hyper parameter 셋 뽑기
        n_splits = 5
        n_repeats = 1
        kf = KFold(n_splits=n_splits, shuffle=True)
        
        X_data = self.X
        Y_data = self.y
        
        CAT_params = {
            "iterations" : [1000, 2000, 3000],
            "depth" : [2, 3],
            "learning": [0.01, 0.05, 0.1],
        }
        
        CAT_list = [[iterations, depth, learning] for iterations in CAT_params["iterations"] for depth in CAT_params["depth"] for learning in CAT_params["learning"]]
        list_ = CAT_list
        
        r2_mean_l, mse_mean_l = [], []
        for model_param in list_ :
            print(model_param)
            r2_list = []
            mse_list = []
            for _ in range(n_repeats):
                # 데이터를 KFold로 분할하여 5번 반복
                for train_index, test_index in kf.split(pd.concat([X_data, Y_data], axis = 1)):
                    
                    if self.obj == "regression":
                        model = CatBoostRegressor(iterations=model_param[0], depth=model_param[1], learning_rate=model_param[2], loss_function='RMSE')
                    elif self.obj == "classfication":
                        model = CatBoostClassifier(iterations=model_param[0], depth=model_param[1], learning_rate=model_param[2])
                        
                    X_train, X_test = X_data.iloc[train_index], X_data.iloc[test_index]
                    y_train, y_test = Y_data.iloc[train_index], Y_data.iloc[test_index]
                    
                    train_pool = Pool(X_train, y_train, ) #cat_features=[0,2,5]
                    test_pool = Pool(X_test)
                    
                    model.fit(train_pool) #fit(X_train, y_train)
                    y_pred = model.predict(test_pool) #predict(X_test)
                    
                    r2_list.append(r2_score(y_test, y_pred))
                    mse_list.append(np.sqrt(mean_squared_error(y_test, y_pred)))
            
            r2_mean = np.mean(r2_list)
            mse_mean = np.mean(mse_list)
            print(r2_mean, ": mean R2 score")
            print(mse_mean, ": mean RMSE score")
            
            r2_mean_l.append([r2_mean, model_param])
            mse_mean_l.append([mse_mean, model_param])
        
        r2_best = r2_mean_l[np.argmax([x[0] for x in r2_mean_l])][1]
        mse_best = mse_mean_l[np.argmin([x[0] for x in mse_mean_l])][1]
        
        if r2_best == mse_best:
            return r2_best
        else:
            return r2_best  # or mse_best
    
    def make_catboost(self, grid_search = True):
        if grid_search:
            best_param = self.Grid_Search_CATBoost()
            self.iter = best_param[0]
            self.N_sub = int(self.iter*0.4)
            self.depth = best_param[1]
            self.lr = best_param[2]
        
        if self.obj == "regression":
            self.catboost = CatBoostRegressor(iterations= self.iter, depth= self.depth, learning_rate= self.lr, loss_function='RMSE')
        elif self.obj == "classfication":
            self.catboost = CatBoostClassifier(iterations= self.iter, depth= self.depth, learning_rate= self.lr)
        #train_pool = Pool(data=self.X, label=self.y)
        self.catboost.fit(self.X, self.y)
        y_pred = self.catboost.predict(self.X) #predict(X_test)
        print("fitted score as R2: ", r2_score(self.y, y_pred), "RMSE: ", np.sqrt(mean_squared_error(self.y, y_pred)))
        
        self.catboost.save_model(fr'{self.catboost_dir}\model.json', format="json")

        try:
            with open(fr'{self.catboost_dir}\model.json', 'r', encoding="utf-8") as file:
                self.cat_json = json.load(file)
                self.bias = self.cat_json["scale_and_bias"][1][0]
                self.scale = self.cat_json["scale_and_bias"][0]
        except:
             with open(fr'{self.catboost_dir}\model.json', 'r', encoding="ISO-8859-1") as file:
                self.cat_json = json.load(file)
                self.bias = self.cat_json["scale_and_bias"][1][0]
                self.scale = self.cat_json["scale_and_bias"][0]
                
        print("CATboost submit complete")
    
    # CATboost 에 있는 tree 들에 대해 clustering 하는 코드
    # k = 7 정도로 설정해두고, 사용한 feature 종류에 따라 분리
    
    def clustering_catboost(self):
        k = self.k
        tree_data = []
        for tree_idx, tree in enumerate(self.cat_json['oblivious_trees']):
            splits = tree['splits']
            leaf_values = tree['leaf_values']
            used_features = set(split['float_feature_index'] for split in splits)
            
            tree_data.append({
                'tree_index': tree_idx,
                'leaf_values': leaf_values,
                'used_features': used_features,
                'split_conditions': [(split['float_feature_index'], split['border']) for split in splits]
            })
            
        df_tree = pd.DataFrame(tree_data)
        mlb = MultiLabelBinarizer()
        feature_matrix = mlb.fit_transform(df_tree['used_features'])  # 각 트리에서 사용된 피처를 One-Hot으로 변환
        
        n_clusters = k  # 클러스터 수 설정
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(feature_matrix)

        df_tree['cluster'] = clusters
        
        self.df_tree = df_tree
        print(f"clustering finished with k = {k}")
        
        return self.df_tree
    
    # jth cluster에서 우선 사용된 전체 feature를 확인하고, cluster 안에 소속된 tree들을 random 하게 호출해서 data를 하나씩 만드는 과정
    # 여기서 만들어진 synthetic data는 origin data와 합쳐서 하나의 작은 NN을 학습시키는데 사용됨
    def DataGeneration_cluster(self, j = 0, J_sub = 0.6, N_data = 1000, sampling = "gibbs"):
        jth_cluster = self.df_tree.loc[self.df_tree["cluster"] == j].copy()
        jth_features = []
        for features in jth_cluster["used_features"]:
            for feature in list(features):
                if feature not in jth_features:
                    jth_features.append(feature)
        
        jth_leaf_values = []
        for leaf_values in jth_cluster["leaf_values"]:
            jth_leaf_values = jth_leaf_values + leaf_values
        jth_leaf_values = list(set(jth_leaf_values))
        
        if np.var(jth_leaf_values) <= (len(jth_cluster)/self.iter)**2*np.var(self.y):
            print(f"{j} regard as bagging, {self.k} bins seleciton")
            N_data = self.X.shape[0] * 9
        else :
            print(f"{j} regard as boosting, random leaf selection")
            N_data = self.X.shape[0] * 9
            
        if sampling == "gibbs":
            gibbs_data_list = []
            
            for _ in range(max(min(int(20000/(len(self.X))) - 1, 9), 0)):
                X_values = self.X.sample(n=min(len(self.X), 250))
                gibbs_gen = gibbs_generator(X = X_values.iloc[:, jth_features], iter =3)
                gibbs_gen.model_make()
                for _iter in range(int(len(self.X)/250)):
                    gibbs_data_list.append(gibbs_gen.gibbs_data())
            
            gibbs_data = np.array(pd.concat([self.X.iloc[:, jth_features]] + gibbs_data_list, axis = 0)).reshape(-1,len(jth_features))
            x_samples = gibbs_data.copy()
        
        elif sampling == "tree":
            leaf_bins = pd.cut(jth_leaf_values, bins=self.N_bins, labels=False, retbins=True)[1]
            random_leaf = True
            
            x_samples = np.random.uniform(self.bounds[0][jth_features], self.bounds[1][jth_features], size=(N_data, len(jth_features)))
            
            if random_leaf :
                batch = self.X.shape[0]
                for _ in range(N_data//batch + 1):
                    x_samples_ = x_samples[_*batch:min((_+1)*batch, N_data)].copy()
                    selected_indices = np.random.choice(jth_cluster.index, size= int(J_sub*len(jth_cluster)), replace=False,)
                    for index in selected_indices:
                        x_samples_ = self.DataGeneration_DT(index, len(x_samples_), -1, x_samples_, jth_features = jth_features)
                    x_samples[_*batch:min((_+1)*batch, N_data)] = x_samples_
            else:
                batch = self.X.shape[0]
                for _ in range(self.N_bins):
                    x_samples_ = x_samples[_*batch:min((_+1)*batch, N_data)].copy()
                    selected_indices = np.random.choice(jth_cluster.index, size= int(J_sub*len(jth_cluster)), replace=False,)
                    for index in selected_indices:
                        x_samples_ = self.DataGeneration_DT(index, len(x_samples_), _, x_samples_, leaf_bins, jth_features)
                    x_samples[_*batch:min((_+1)*batch, N_data)] = x_samples_
            
            x_original = np.array([self.X.iloc[:, jth_features]]).reshape(len(self.X), len(jth_features))
            x_samples = np.vstack([x_samples, x_original])
            
            
        def predict(tree, features):
            split_conditions = tree["split_conditions"]
            leaf_values = tree["leaf_values"]
            leaf_index = []
            for (feature, threshold) in list(reversed(split_conditions)):
                if features[feature] <= threshold:
                    leaf_index.append("0")
                else :
                    leaf_index.append("1")
            leaf_index = int("".join(leaf_index), 2)
            
            if self.obj == "regression":
                return leaf_values[leaf_index]
            elif self.obj == "classfication":
                return leaf_values[leaf_index * self.y_class : (leaf_index + 1) * self.y_class]
        
        split_conditions = [row["split_conditions"] for _, row in jth_cluster.iterrows()]
        leaf_values = [row["leaf_values"] for _, row in jth_cluster.iterrows()]
        used_features = [set(row["used_features"]) for _, row in jth_cluster.iterrows()]
        
        predictions = []
        
        # 벡터화된 연산으로 각 샘플에 대해 예측 수행
        for input_features in x_samples:
            self.input_feature_dict = dict(zip(jth_features, input_features))
            
            # 한 번에 트리별 예측값 계산
            x_pred = [
                predict(
                    {"split_conditions": split, "leaf_values": leaf},
                    {k: self.input_feature_dict[k] for k in self.input_feature_dict if k in use_feats}
                )
                for split, leaf, use_feats in zip(split_conditions, leaf_values, used_features)
            ]
            
            if self.obj == "regression":
                predictions.append(np.sum(x_pred))
            elif self.obj == "classfication":
                x_pred = [item if len(item) > 0 else [1/self.y_class for _ in range(self.y_class)] for item in x_pred]
                x_pred = np.array(x_pred)
                class_pred = [np.sum(x_pred[:, _]) for _ in range(self.y_class)]
                predictions.append(class_pred)
                
        jth_data = pd.DataFrame(x_samples, columns = self.X.columns[jth_features])
        
        if self.obj == "regression":
            jth_data["label"] = pd.Series(predictions)
        elif self.obj == "classfication":
            predictions = np.array(predictions)
            for _ in range(self.y_class):
                jth_data[f"{_}_prob"] = pd.Series(predictions[:, _])
        
        return jth_data, jth_features
    
    def DataGeneration_DT(self, tree_idx, n_samples=1, j = 0, x_samples= [], leaf_bins = np.array([]), jth_features = []):
        
        tree = self.df_tree.iloc[tree_idx]
        used_features = list(tree["used_features"])
        split_conditions = list(reversed(tree["split_conditions"]))
        leaf_values = tree["leaf_values"]
        
        integer_idxs = [jth_features.index(col) for col in [idx for idx, c_col in enumerate(self.X.columns) if (("D" in c_col) or ("B" in c_col)) and (idx in used_features)]]
        onehot_idxs = [jth_features.index(col) for col in [idx for idx, c_col in enumerate(self.X.columns) if ("O" in c_col) and (idx in used_features)]]
        
        leaf_idx = -1
        if j >= 0 and len(leaf_bins) > 0 :
            for idx, leaf_value in enumerate(leaf_values):
                if leaf_value >= leaf_bins[j] and leaf_value < leaf_bins[j+1]:
                    leaf_idx = idx
                    break
            if leaf_idx < 0 :
                leaf_idx = np.random.randint(0, len(leaf_values))
        else :
            leaf_idx = np.random.randint(0, len(leaf_values))
            
        zfill = math.ceil(math.log2(leaf_idx + 1))
        binary = bin(leaf_idx)[2:]
        binary = binary.zfill(zfill)
        for _ in range(n_samples):
            n = 0
            for direction in binary:
                # direction 0이면 왼쪽, 1이면 오른쪽
                # threshold가 feature 최솟값보다 작거나, 최댓값보다 큰 경우도 있어 해당 경우 보정
                feature_to_split = split_conditions[n][0]
                x_samples_idx = jth_features.index(feature_to_split)
                
                if direction == 0 :
                    if x_samples[_][x_samples_idx] > split_conditions[n][1]:
                        low, high = self.bounds[0][feature_to_split], max(self.bounds[0][feature_to_split], split_conditions[n][1])
                        x_samples[_][x_samples_idx] = np.random.uniform(low, high)
                        #x_samples[_][feature_split] = np.random.uniform(self.bounds[0][feature_split], threshold[n])
                    n += 1
                else:
                    if x_samples[_][x_samples_idx] <= split_conditions[n][1]:
                        feature_idx = split_conditions[n][0]
                        low, high = min(self.bounds[1][feature_to_split], split_conditions[n][1]), self.bounds[1][feature_to_split]
                        x_samples[_][x_samples_idx] = np.random.uniform(low, high)
                        # x_samples[_][feature_split] = np.random.uniform(threshold[n], self.bounds[1][feature_split])
                    n += 1
        
        ## Discrete한 feature의 경우 기존 정수 범위 보다 0.5 만큼 +로 shift 한 다음, 마지막에 return할 때 int() 씌워서 정수 처리 구현
        ## 즉, 0.5 ~ 1.5 에서 버림을 한다면, [0.5, 1.0) ==> 0, [1.0, 1.5] ==> 1이 됨. 각 범위에 떨어질 확률은 uniform 하므로 1/2.
            if integer_idxs != []:
                x_samples[_][integer_idxs] = (x_samples[_][integer_idxs] + 0.5).astype(int)
            if onehot_idxs != []:
                x_samples_c = x_samples.copy()
                x_samples[_][onehot_idxs] = 0
                x_samples[_][onehot_idxs[np.argmax(x_samples_c[_][onehot_idxs])]] = 1
            
        return x_samples

    # Dynamic_Selection에서 나온 데이터와 기존 Original Data를 특정 비율로 concat 후 반환
    # 데이터 셋 concat하고, normalize -1 ~ 1
    def Concat_Data(self, N_data = 1000):
        #DT_distribution = self.Find_Weight()

        synthetic_data = {}
        for j in range(self.k):
            jth_data, jth_features = self.DataGeneration_cluster(j)
            synthetic_data[j] = {"data" : jth_data, "features": jth_features}

        self.synthetic_data = synthetic_data
        print("data generation completed")