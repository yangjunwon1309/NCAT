import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, \
    accuracy_score, f1_score
from multiprocessing import Manager, Process
import json

from .NN_model import adam_train, softmax, DNN_LM
from .NCAT_datageneration import NCAT_datageneration, gibbs_generator


def train_single_nn(idx, data, label, rank, y_rmse, obj, cluster_k, max_iter, min_iter, lr, loss_type):
    if rank == 2:
        hidden_dim_list = [8, 4]
        max_iter = min(int((7 / cluster_k) * ((data.shape[1] * 20) + 20)) * 100, max_iter)
        tolerance_list = [0.95, y_rmse / 3]
    elif rank == 1:
        hidden_dim_list = [4, 4]
        max_iter = min(int((7 / cluster_k) * ((data.shape[1] * 20) + 15)) * 100, max_iter)
        tolerance_list = [0.9, y_rmse / 3]
    else:
        hidden_dim_list = [4, 4]
        max_iter = min(int((7 / cluster_k) * ((data.shape[1] * 20) + 15)) * 100, max_iter)
        tolerance_list = [0.9, y_rmse / 2]

    if obj == "classification":
        tolerance_list = [0.9, 0.09]

    batch_size_ = min(32, int(data.shape[0] / 10))

    print(f"{idx}th NN multi proccessed as : ", rank, hidden_dim_list, max_iter, tolerance_list)

    nn_model = DNN_LM(input_dim=data.shape[1], hidden_dim=hidden_dim_list[0], hidden_dim_2=hidden_dim_list[1],
                      output_dim=label.shape[1], obj=obj)
    adam_train(nn_model, data, label, batch_size=batch_size_, tolerance_r2=tolerance_list[0],
               tolerance_rmse=tolerance_list[1], max_iterations=max_iter, min_iterations=min_iter, lr=lr, beta1=0.9, beta2=0.999,
               epsilon=1e-8, weight_decay=0.0, loss_type=loss_type, multi=True)
    print(f"{idx}th NN train finshed")

    return nn_model


class NCAT():
    def __init__(self, X=np.array([]), y=np.array([]), cluster_k=7, obj="regression", max_iter=50000, min_iter=5000,
                 N_bins=10):
        self.X = X
        self.y = y
        self.scaler = MinMaxScaler()
        self.cluster_k = cluster_k
        self.scaled = False
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.bias = 0
        self.scale = 1
        self.N_bins = N_bins

        if obj not in ["regression", "classfication"]:
            return "invalid objective, please input as regression either classfication"

        self.obj = obj
        self.catboost_dir = os.path.join(os.getcwd().replace("\\", "/"), f'NCAT_catboost_{self.obj}')

        if self.obj == "classfication":
            self.y_class = len(pd.Series(self.y).unique())

    def pipeline(self, X):
        # minmax scaler 로 transform 하는 것만 일단 존재, predict에서 이후 feature 골라서 처리 함
        # 반드시 feature 들의 column 순서가 처음과 같아야 함
        if self.scaled:
            assert self.X.columns == X.columns

            scaled_data = pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)

            return scaled_data
        else:
            print("please fit the training set and distill catboost first")
            return None

    def scale_and_split(self):
        data = pd.concat([self.X], axis=1)
        test_size = 0.1

        self.scaler.fit(data)
        scaled_data = pd.DataFrame(self.scaler.transform(data), columns=data.columns, index=data.index)
        scaled_data = pd.concat([scaled_data, self.y], axis=1)

        train_data = scaled_data.sample(frac=1 - test_size, random_state=None)
        test_data = scaled_data.drop(train_data.index)

        # Separate X and y for train and test sets
        X_train = train_data[self.X.columns]
        y_train = train_data[scaled_data.columns[-1]]
        X_test = test_data[self.X.columns]
        y_test = test_data[scaled_data.columns[-1]]

        return X_train, X_test, y_train, y_test

    # Data_Generator 로 imitation 위한 데이터 생성
    # NN 위해 data 전체에 대해서 전처리 진행
    # feature의 경우 -1 ~ 1 로 Normal, label은 ??
    def data_generator(self, scale_and_split=True):
        X_train, X_test, y_train, y_test = self.scale_and_split()
        self.X_test = np.array(X_test)
        self.X_train = np.array(X_train)
        self.y_test = y_test
        self.y_train = y_train

        self.ncat = NCAT_datageneration(X_train, y_train, iter=1500, depth=2, lr=0.05, k=self.cluster_k, obj=self.obj,
                                        N_bins=self.N_bins)
        self.ncat.make_catboost(False)
        self.ncat.clustering_catboost()
        self.ncat.Concat_Data()

        self.bias = self.ncat.bias
        self.scale = self.ncat.scale

        synthetic_data = self.ncat.synthetic_data

        cluster_syn_data = {}

        if self.obj == "regression":
            for _ in range(self.cluster_k):
                data = np.array(synthetic_data[_]["data"].drop(columns=["label"]))
                label = np.array(synthetic_data[_]["data"]["label"])
                features = synthetic_data[_]["features"]

                if len(label.shape) == 1:
                    label = label[:, np.newaxis]

                cluster_syn_data[_] = {"X": data, "y": label, "features": features}
        else:
            for _ in range(self.cluster_k):
                prob_col = [col for col in synthetic_data[_]["data"].columns if "prob" in col]
                data = np.array(synthetic_data[_]["data"].drop(columns=prob_col))
                label = np.array(synthetic_data[_]["data"][prob_col])
                features = synthetic_data[_]["features"]

                if label.shape[1] == 1:
                    label = label[:, np.newaxis]

                cluster_syn_data[_] = {"X": data, "y": label, "features": features}

        self.cluster_syn_data = cluster_syn_data
        self.scaled = True

        return cluster_syn_data

    def distillation(self, lr=0.07):
        cluster_syn_data = self.data_generator()
        self.NN_dict = {}

        for _ in range(self.cluster_k):
            label = cluster_syn_data[_]["y"]
            cluster_syn_data[_]["y_rmse"] = np.sqrt(np.var(label))
            cluster_syn_data[_]["y_stat"] = np.mean(label) ** 2 + np.var(label) + (np.max(label) - np.min(label)) ** 2

        sorted_cluster_syn_data = sorted(
            cluster_syn_data.values(),
            key=lambda x: x["y_stat"],
            reverse=True)

        total_count = len(sorted_cluster_syn_data)

        top_75_idx = int(total_count * 0.75)
        bottom_25_idx = int(total_count * 0.25)

        for idx, cluster in enumerate(sorted_cluster_syn_data):
            if idx < bottom_25_idx:
                cluster["rank"] = 0
            elif idx < top_75_idx:
                cluster["rank"] = 1
            else:
                cluster["rank"] = 2

        self.cluster_syn_data = sorted_cluster_syn_data

        default_max_iter = self.max_iter
        if self.obj == "regression":
            loss_type = "mse"
        else:
            loss_type = "mse_class"  # "xentropy"

        for _ in range(self.cluster_k):
            data = sorted_cluster_syn_data[_]["X"]
            label = sorted_cluster_syn_data[_]["y"]
            rank = sorted_cluster_syn_data[_]["rank"]
            y_rmse = sorted_cluster_syn_data[_]["y_rmse"]

            if rank == 2:
                hidden_dim_list = [8, 4]
                max_iter = min(int((7 / self.cluster_k) * ((data.shape[1] * 10) + 20)) * 100, default_max_iter)
                tolerance_list = [0.95, y_rmse / 3]
            elif rank == 1:
                hidden_dim_list = [4, 4]
                max_iter = min(int((7 / self.cluster_k) * ((data.shape[1] * 15) + 15)) * 100, default_max_iter)
                tolerance_list = [0.9, y_rmse / 3]
            else:
                hidden_dim_list = [4, 4]
                max_iter = min(int((7 / self.cluster_k) * ((data.shape[1] * 12.5) + 15)) * 100, default_max_iter)
                tolerance_list = [0.9, y_rmse / 2]

            if self.obj == "classfication":
                tolerance_list = [0.9, 0.09]

            batch_size_ = min(32, int(data.shape[0] / 10))

            print(f"{_}th NN: ", rank, hidden_dim_list, max_iter, tolerance_list)
            self.NN_dict[_] = DNN_LM(input_dim=data.shape[1], hidden_dim=hidden_dim_list[0],
                                     hidden_dim_2=hidden_dim_list[1], output_dim=label.shape[1])
            adam_train(self.NN_dict[_], data, label, batch_size=batch_size_, tolerance_r2=tolerance_list[0],
                       tolerance_rmse=tolerance_list[1], max_iterations=max_iter, lr=lr, beta1=0.9, beta2=0.999,
                       epsilon=1e-8, weight_decay=0.0, loss_type=loss_type)

    def distillation_multi(self, lr=0.01):
        cluster_syn_data = self.data_generator()
        self.NN_dict = {}

        # 클러스터 데이터 준비
        for _ in range(self.cluster_k):
            label = cluster_syn_data[_]["y"]
            cluster_syn_data[_]["y_rmse"] = np.sqrt(np.var(label))
            cluster_syn_data[_]["y_stat"] = np.mean(label) ** 2 + np.var(label) + (np.max(label) - np.min(label)) ** 2

        sorted_cluster_syn_data = sorted(
            cluster_syn_data.values(),
            key=lambda x: x["y_stat"],
            reverse=True
        )

        total_count = len(sorted_cluster_syn_data)
        top_75_idx = int(total_count * 0.75)
        bottom_25_idx = int(total_count * 0.25)

        for idx, cluster in enumerate(sorted_cluster_syn_data):
            if idx < bottom_25_idx:
                cluster["rank"] = 0
            elif idx < top_75_idx:
                cluster["rank"] = 1
            else:
                cluster["rank"] = 2

        self.cluster_syn_data = sorted_cluster_syn_data

        max_iter = self.max_iter
        min_iter = self.min_iter
        loss_type = "mse" if self.obj == "regression" else "mse_class"

        # 멀티프로세싱을 통해 병렬 처리
        processes = []
        with Manager() as manager:
            shared_dict = manager.dict(self.NN_dict)
            for idx in range(self.cluster_k):
                cluster = sorted_cluster_syn_data[idx]
                data = cluster["X"]
                label = cluster["y"]
                rank = cluster["rank"]
                y_rmse = cluster["y_rmse"]

                process = Process(target=self.train_single_nn_and_store, args=(
                idx, data, label, rank, y_rmse, shared_dict, self.obj, self.cluster_k, max_iter, min_iter, lr,
                loss_type))
                processes.append(process)
                process.start()

            # 모든 프로세스가 완료될 때까지 대기
            for process in processes:
                process.join()

            self.NN_dict = dict(shared_dict)
            print("All NNs are trained in parallel!")

    def train_single_nn_and_store(self, idx, data, label, rank, y_rmse, shared_dict, obj, cluster_k, max_iter, min_iter,
                                  lr, loss_type):
        # train_single_nn 호출idx, data, label, rank, y_rmse, obj, cluster_k, max_iter, lr, loss_type
        nn_model = train_single_nn(idx, data, label, rank, y_rmse, obj, cluster_k, max_iter, min_iter, lr, loss_type)

        shared_dict[idx] = nn_model

    def testing(self):
        if self.obj == "regression":
            predictions = self.predict(self.X_test)
            r2 = r2_score(self.y_test, predictions)
            rmse = np.sqrt(mean_squared_error(self.y_test, predictions))
            matric_list = [r2, rmse]

        else:
            predictions = self.predict(self.X_test)
            acc = accuracy_score(self.y_test, predictions),
            f1 = f1_score(self.y_test, predictions, average='macro', ),
            r2 = r2_score(self.y_test, predictions)
            rmse = np.sqrt(mean_squared_error(self.y_test, predictions))
            matric_list = [acc, f1, r2, rmse]

        return matric_list, predictions

    def get_params(self, j=-1):
        if j > -1:
            params = self.NN_dict[j].get_params()
            return params

        params_list = []
        for _ in range(len(self.NN_dict)):
            params_list.append(self.NN_dict[_].get_params().ravel())
        whole_params = np.hstack(params_list)
        return whole_params

    def set_params(self, params, j=-1):
        # 그냥 assert를 나중에 박아놓는게...
        if j > -1:
            try:
                self.NN_dict[j].set_params(params)
            except Exception as e:
                print("canont set params to jth NN as", e)
        else:
            pivot = 0
            for idx in range(len(self.NN_dict)):
                jth_param_len = len(self.NN_dict[idx].get_params())
                self.NN_dict[idx].set_params(params[pivot:pivot + jth_param_len])
                pivot += jth_param_len

    def predict(self, X_test=np.array([])):
        if X_test.shape[0] < 1:
            return "no X"

        if self.obj == "regression":
            predictions = [0 for _ in range(X_test.shape[0])]
            for _ in range(self.cluster_k):
                features = self.cluster_syn_data[_]["features"]
                pred = list(self.NN_dict[_].predict(X_test[:, features]).reshape(X_test.shape[0], 1))
                predictions = [(i + pred_i) for i, pred_i in zip(predictions, pred)]
            predictions = [self.bias + pred_i * self.scale for pred_i in predictions]
            predictions = np.array([pred[0] for pred in predictions]).reshape(-1, 1)
        else:
            predictions = [np.array([0 for _ in range(self.y_class)]) for _ in range(X_test.shape[0])]
            for _ in range(self.cluster_k):
                features = self.cluster_syn_data[_]["features"]
                pred = self.NN_dict[_].predict(X_test[:, features]).reshape(X_test.shape[0], self.y_class)
                predictions = [pred_x + predict_x for pred_x, predict_x in zip(pred, predictions)]
            predictions = [np.argmax(softmax(self.bias + pred_i * self.scale)) for pred_i in predictions]

        return predictions

    def save(self, file_name="model_info"):
        model_info = {}
        for _ in range(self.cluster_k):
            jth_nn = self.NN_dict[_]
            jth_param = jth_nn.get_params()
            jth_dim = [jth_nn.input_dim, jth_nn.hidden_dim, jth_nn.hidden_dim_2, jth_nn.output_dim]
            jth_activate_func = [jth_nn.layer1, jth_nn.layer2]
            used_features = self.cluster_syn_data[_]["features"]
            model_info[_] = {"param": jth_param,
                             "dimension": jth_dim,
                             "activate function": jth_activate_func,
                             "features": used_features}

        model_info["bias_and_scale"] = [self.bias, self.scale]
        model_info["whole_param"] = self.get_params()
        model_info["data_info"] = {"obj": self.obj, "cluster_k": self.cluster_k,
                                   "max_iter": self.max_iter, "catboost_dir": self.catboost_dir} if (
                    self.obj == "regression") else {"obj": self.obj, "cluster_k": self.cluster_k,
                                                    "max_iter": self.max_iter, "catboost_dir": self.catboost_dir,
                                                    "y_class": self.y_class}

        def save_scaler_to_json(scaler):
            if not isinstance(scaler, MinMaxScaler):
                raise ValueError("Provided scaler must be an instance of MinMaxScaler.")

            # Scaler 파라미터 추출
            return {
                "min_": scaler.min_.tolist(),
                "scale_": scaler.scale_.tolist(),
                "data_min_": scaler.data_min_.tolist(),
                "data_max_": scaler.data_max_.tolist(),
                "data_range_": scaler.data_range_.tolist(),
                "feature_range": scaler.feature_range
            }

        model_info["scaler"] = save_scaler_to_json(self.scaler)

        def serialize_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()  # numpy 배열을 Python 리스트로 변환
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        def save_model_info_to_json(model_info, file_path=f"{self.catboost_dir}/{file_name}.json"):
            try:
                with open(file_path, "w") as json_file:
                    json.dump(model_info, json_file, indent=4, default=serialize_numpy)  # indent=4는 보기 좋게 정렬
                print(f"Model information successfully saved to {file_path}")
            except Exception as e:
                print(f"An error occurred while saving the model information: {e}")

        save_model_info_to_json(model_info)

    def load(self, file_name="model_info", file_dir=""):
        self.NN_dict = {}
        self.cluster_syn_data = {}
        file_path = fr"{self.catboost_dir}/{file_name}.json"
        if len(file_dir) > 0:
            file_path = file_dir
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")

        try:
            with open(file_path, "r") as json_file:
                data = json.load(json_file)
            print(f"File '{file_path}' successfully loaded.")

        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON: {e}")
            return
        except Exception as e:
            print(f"An error occurred while reading the JSON file: {e}")
            return

        self.bias, self.scale = data["bias_and_scale"][0], data["bias_and_scale"][1]
        self.cluster_k = data["data_info"]["cluster_k"]

        def load_scaler_from_json(scaler_info):

            scaler = MinMaxScaler(feature_range=tuple(scaler_info["feature_range"]))
            scaler.min_ = np.array(scaler_info["min_"])
            scaler.scale_ = np.array(scaler_info["scale_"])
            scaler.data_min_ = np.array(scaler_info["data_min_"])
            scaler.data_max_ = np.array(scaler_info["data_max_"])
            scaler.data_range_ = np.array(scaler_info["data_range_"])

            print(f"Scaler settings loaded from {file_path}")
            return scaler

        self.scaler = load_scaler_from_json(data["scaler"])

        if data["data_info"]["obj"] == "classfication":
            self.y_class = data["data_info"]["y_class"]

        for nn_idx in [int(i) for i in data.keys() if
                       i not in ["whole_param", "bias_and_scale", "scaler", "data_info"]]:
            dimension = data[str(nn_idx)]["dimension"]
            param = data[str(nn_idx)]["param"]
            features = data[str(nn_idx)]["features"]
            layers = data[str(nn_idx)]["activate function"]

            self.NN_dict[nn_idx] = DNN_LM(input_dim=dimension[0], hidden_dim=dimension[1], hidden_dim_2=dimension[2],
                                          output_dim=dimension[3], layer1=layers[0], layer2=layers[1])
            self.NN_dict[nn_idx].set_params(np.array(param))
            self.cluster_syn_data[nn_idx] = {"features": features}


def online_finetune(NCAT, data=[], label=[], max_iter=100, batch_size=16, lr=0.01, ):
    # 전체 예측 결과에 대한 loss를 기준으로 각 j th NN들을 모두 튜닝해야 함
    assert len(data) == len(label)

    if (len(data) < 1) or (len(label) < 1):
        data = NCAT.X_test
        label = np.array(NCAT.y_test).reshape(-1, 1)

    data = NCAT.pipeline(data)

    tolerance_list = [0.95, 0.8 * np.sqrt(np.var(label))]
    loss_type = "MSE"

    if NCAT.obj == "classfication":
        loss_type = "xentropy"
    elif NCAT.obj == "regression":
        loss_type = "MSE"

    adam_train(NCAT, data, label, batch_size=batch_size, tolerance_r2=tolerance_list[0],
               tolerance_rmse=tolerance_list[1], max_iterations=max_iter, lr=lr, beta1=0.9, beta2=0.999, epsilon=1e-8,
               weight_decay=0.0, loss_type=loss_type)
    print("online tuning success")

    return NCAT