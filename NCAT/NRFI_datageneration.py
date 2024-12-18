

class DataGeneration_RF():
    
    # RF 기본 hyperparameter 구현
    # 기타 기본 설정 구현 (N_sub, )
    # imitation_NN class에서 X, y 를 X_train, y_train으로 쪼개서 해당 class로 전달할 예정. 이 클래스에서는 RF를 생성하고, imitation을 위한 데이터 생성 하는 역할 수행
    def __init__(self, X_train, y_train, N = 500, max_d = 10):
        self.X = self.auto_columns(X_train) # X_train
        self.y = y_train
        self.N = N
        self.N_sub = int(self.N*0.4)
        self.max_d = max_d
        self.max_fn = int(np.sqrt(self.X.shape[1]))
        self.N_bins = 10
        self.bounds = self.boundary(self.N_bins)
        
        #self.model = RandomForestRegressor(n_estimators = self.N, max_depth=self.max_d, max_features=self.max_fn,  bootstrap=True, criterion='squared_error',)
        self.model = RandomForestRegressor(n_estimators = self.N, max_depth=self.max_d, max_features= 1.0, criterion='squared_error', min_samples_split = 5)
        self.Generated_Data = pd.DataFrame()
    
    
    # X_train에서 categorical -> one hot encoding으로 변경
    # one hot encoding과 같이 discrete 한 경우 => D_i 식으로 열 이름 변경
    # continuous 한 경우 ==> C_i 식으로 열 이름 변경
    def auto_columns(self, X):
        X_return = X.copy()
        
        # int, float 열들에 대해서 새롭게 이름 지정
        integer_columns = [
            col for col in X.columns
            if X[col].apply(lambda x: isinstance(x, (int, float)) and x == int(x)).all()
        ]
        string_columns = [col for col in X.columns if X[col].apply(lambda x: isinstance(x, str)).all()]
        continuous_columns = [[col, f"C_{i}"] for i, col in enumerate([c_col for c_col  in X.columns if (c_col not in integer_columns + string_columns)])]
        integer_columns = [[col, f"D_{i}"] for i, col in enumerate(integer_columns)]
        
        # String 열들에 대해서 one-hot-encoding 열 생성 (int)
        if string_columns:
            X_encoded = pd.get_dummies(X, columns=string_columns, drop_first=False)
            new_columns = list(set(X_encoded.columns) - set(X.columns))
            X[new_columns] = X_encoded[new_columns].astype(int)
        else:
            new_columns = []

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
            [col[1] for col in one_hot_columns]
        ]
    
    def boundary(self, n_bins = 10):
        continuous_columns = [c_col for c_col in self.X.columns if "C" in c_col]
        integer_columns = [c_col for c_col in self.X.columns if ("D" in c_col) or ("O" in c_col)]
        
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
    def Make_RF(self):
        self.model.fit(self.X, self.y)
        
    
    # 각 DT에서 data sample 만드는 코드
    # 이후 Dynamic_Selection에서 호출해서 데이터 생성
    def DataGeneration_DT(self, tree, n_samples=1, j = 0, x_samples= []):
        if len(x_samples) < 1:
            x_samples = np.random.uniform(self.bounds[0], self.bounds[1], size=(n_samples, self.X.shape[1]))
        integer_idxs = [idx for idx, c_col in enumerate(self.X.columns) if "D" in c_col]
        onehot_idxs = [idx for idx, c_col in enumerate(self.X.columns) if "O" in c_col]
        
        def calculate_node_weights(n_nodes, children_left, children_right, value):
            # 노드별 weight 초기화
            n_bins = self.N_bins
            node_weights = np.zeros((n_nodes, n_bins))

            # 리프 노드에 대한 bin 확률 계산
            leaf_indices = np.where(children_left == children_right)[0]
            y_bin = self.y_bin()
            
            for leaf in leaf_indices:
                leaf_value = value[leaf][0][0]    
                bin_index = self.N_bins - 1
                for j in range(n_bins):
                    if y_bin[j] <= leaf_value and y_bin[j+1] > leaf_value:
                        bin_index = j
                        break

                node_weights[leaf][bin_index] = 1

            # 리프에서 루트로 확률 전파
            def propagate_weights(node):
                if children_left[node] == children_right[node]:  # 리프 노드
                    return node_weights[node]
                
                left_weights = propagate_weights(children_left[node])
                right_weights = propagate_weights(children_right[node])
                
                # 자식 노드의 가중치 합산
                node_weights[node] = left_weights + right_weights
                
                return node_weights[node]

            # 루트 노드부터 시작하여 모든 노드의 가중치 계산
            propagate_weights(0)

            return node_weights


        target_tree = tree #self.RF.estimators_[0].tree_

        n_nodes = target_tree.node_count
        children_left = target_tree.children_left
        children_right = target_tree.children_right
        value = target_tree.value
        feature = target_tree.feature
        threshold = target_tree.threshold
        
        feature_names = self.X.columns.tolist()

        node_weights = calculate_node_weights(n_nodes, children_left, children_right, value) + 0.0000001
        #display(node_weights)
        
        for _ in range(n_samples):
            n = 0
            j = j # 0 ~ 9
            features_remain = list(self.X.columns)
            
            while n >= 0:
                w_left, w_right = node_weights[children_left[n]][j], node_weights[children_right[n]][j]
                try:
                    total = np.add(w_left, w_right, dtype=np.float64)
                    if total > 0:
                        w_left, w_right = w_left / total, w_right / total
                    #print(total, w_left, w_right)
                except Exception as e:
                    print(e)
                    print(total, w_left, w_right, "changed as 0.5, 0.5")
                    w_left, w_right = 0.5, 0.5
                    
                direction = np.random.choice([0, 1], p=[w_left, w_right])
                feature_split = feature[n]
                
                if feature_split in features_remain:
                    features_remain.remove(feature_split)
                else:
                    node_weights[n][j] = max(0, node_weights[n][j] * abs(np.random.normal(0, 3)))
                
                
                # direction 0이면 왼쪽, 1이면 오른쪽
                # threshold가 feature 최솟값보다 작거나, 최댓값보다 큰 경우도 있어 해당 경우 보정
                if direction == 0 :
                    if x_samples[_][feature_split] > threshold[n]:
                        low, high = min(self.bounds[0][feature_split], max(threshold[n], self.bounds[0][feature_split])), max(self.bounds[0][feature_split], threshold[n])
                        x_samples[_][feature_split] = np.random.uniform(low, high)
                        #x_samples[_][feature_split] = np.random.uniform(self.bounds[0][feature_split], threshold[n])
                    n = children_left[n]
                else:
                    if x_samples[_][feature_split] < threshold[n]:
                        low, high = min(self.bounds[1][feature_split], threshold[n]), max(self.bounds[1][feature_split], min(threshold[n], self.bounds[1][feature_split]))
                        x_samples[_][feature_split] = np.random.uniform(low, high)
                        # x_samples[_][feature_split] = np.random.uniform(threshold[n], self.bounds[1][feature_split])
                    n = children_right[n]
        
        ## Discrete한 feature의 경우 기존 정수 범위 보다 0.5 만큼 +로 shift 한 다음, 마지막에 return할 때 int() 씌워서 정수 처리 구현
        ## 즉, 0.5 ~ 1.5 에서 버림을 한다면, [0.5, 1.0) ==> 0, [1.0, 1.5] ==> 1이 됨. 각 범위에 떨어질 확률은 uniform 하므로 1/2.
            if integer_idxs != []:
                x_samples[_][integer_idxs] = (x_samples[_][integer_idxs] + 0.5).astype(int)
            if onehot_idxs != []:
                x_samples_c = x_samples.copy()
                x_samples[_][onehot_idxs] = 0
                x_samples[_][onehot_idxs[np.argmax(x_samples_c[_][onehot_idxs])]] = 1
            
        return x_samples
    
    # Data의 Confidence가 최대한 Uniform 하게 뽑히도록, 가중치 설정
    def Find_Weight(self):
        N_ = self.N
        n_features = self.X.shape[1]
        N_bins = self.N_bins
        bin_edges_per_feature = []
        w = np.ones(N_)
        bin_totals = []
        
        for col in range(n_features):
            min_val, max_val = self.bounds[0][col], self.bounds[1][col]
            # min_val과 max_val이 동일한 경우, bin_edges를 적절히 설정
            if min_val == max_val:
                bin_edges = np.linspace(min_val - 0.5, max_val + 0.5, N_bins + 1)
            else:
                bin_edges = np.linspace(min_val, max_val, N_bins + 1)
            bin_edges_per_feature.append(bin_edges)
        
        
        for _ in range(N_):
            hist_counts = np.zeros((n_features, N_bins), dtype=int)
            target_tree = self.model.estimators_[_].tree_
            sampled_x = self.DataGeneration_DT(target_tree, 150, 0)
            
            for col in range(n_features):
                col_data = sampled_x[:, col]
                bin_edges = bin_edges_per_feature[col]
                counts, _ = np.histogram(col_data, bins=bin_edges)
                hist_counts[col] += counts
                
            bin_total = hist_counts.sum(axis = 0)
            bin_totals.append(bin_total)
        
        bin_totals = np.array(bin_totals)
        normalized_bin_totals = bin_totals / np.sum(bin_totals, axis=0, keepdims=True)
        #display(bin_totals)
        #display(normalized_bin_totals)
        def objective(w):
            """최소화할 목적 함수: 모든 feature index i에 대해 σ_j (w_j * h_i,j) - 1의 제곱합."""
            residuals = np.sum(w[:, np.newaxis] * normalized_bin_totals, axis=0) - 1
            return np.sum(residuals**2)  # 제곱합을 최소화

        # 초기값 및 제약 조건 설정
        initial_w = np.ones(N_) / N_  # 초기 가중치 (평균 분포)
        bounds = [(0, 1) for _ in range(N_)]  # 가중치는 [0, 1] 범위
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # 가중치 합은 1

        # 최적화 수행
        result = minimize(objective, initial_w, bounds=bounds, constraints=constraints)

        # 최적화 결과 반환
        if result.success:
            w = result.x  # 최적 가중치
        else:
            raise ValueError("Optimization failed!")
        
        #display(w)
        epsilon = (1 / N_) * 10**-35  # 작은 값 설정
        w = np.maximum(w, epsilon)  # w의 최소값을 epsilon으로 설정
        w = w / np.sum(w)  
        
        return w

    
    # Find_Weight에서 나온 가중치를 바탕으로 N_sub 개 만큼 DT를 설정하고, 데이터 생성
    def Dynamic_Selection(self, N_data = 1000):
        DT_distribution = self.Find_Weight()
        
        trees_sample = []
        target_size = (self.X.shape[0] * 9) // self.N_bins
        for j in range(self.N_bins):
            selected_indices = np.random.choice(self.N, size=self.N_sub, replace=False, p=DT_distribution)
            for idx, selected_ in enumerate(selected_indices):
                target_tree = self.model.estimators_[selected_].tree_
                if idx == 0:
                    jth_tree_sample = self.DataGeneration_DT(target_tree, target_size, j)
                else:
                    jth_tree_sample = self.DataGeneration_DT(target_tree, target_size, j, jth_tree_sample)
            
            trees_sample.append(jth_tree_sample)
        
        samples = np.vstack(trees_sample)
        return samples
    
    # Dynamic_Selection에서 나온 데이터와 기존 Original Data를 특정 비율로 concat 후 반환
    # 데이터 셋 concat하고, normalize -1 ~ 1
    def Concat_Data(self):
        synthetic_data = self.Dynamic_Selection()
        origin_data = np.array(self.X)
        
        concat_data = np.vstack([origin_data, synthetic_data])
        concat_data = pd.DataFrame(concat_data, columns=self.X.columns)
        
        y_label = self.model.predict(concat_data)
        
        # y_label에서 그냥 기존 X_train 값은 실제 값으로 넣어서 학습시키면 어떨까
        
        y_label_syn = self.model.predict(pd.DataFrame(synthetic_data, columns=self.X.columns))[:,np.newaxis]
        y_label_org = np.array(self.y)[:,np.newaxis]
        
        return concat_data, y_label #np.vstack([y_label_syn, y_label_org])