import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, accuracy_score, f1_score

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    # Clipping to avoid overflow
    x = np.clip(x, -709, 709)
    return 1 / (1 + np.exp(-x))

def leaky_relu(x):
    return np.maximum(0.1*x, x)

class DNN_LM:
    def __init__(self, input_dim, hidden_dim, hidden_dim_2, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_dim_2 = hidden_dim_2
        self.output_dim = output_dim
        self.w1 = np.random.randn(input_dim, hidden_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.w2 = np.random.randn(hidden_dim, hidden_dim_2)
        self.b2 = np.zeros((1, hidden_dim_2))
        self.w3 = np.random.randn(hidden_dim_2, output_dim)
        self.b3 = np.zeros((1, output_dim))
        self.w4 = np.random.randn(hidden_dim_2, output_dim)
        self.b4 = np.zeros((1, output_dim))
        self.layer1 = "relu"
        self.layer2 = "leaky_relu"
        
    def forward(self, X):
        self.z1 = X @ self.w1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.w2 + self.b2
        self.a2 = leaky_relu(self.z2)
        self.z3 = self.a2 @ self.w3 + self.b3
        # self.a3 = leaky_relu(self.z3)
        # self.z4 = self.a3 @ self.w4 + self.b4
        return self.z3

    
    def predict(self, X):
        return self.forward(X)

    def get_params(self):
        return np.hstack([self.w1.ravel(), self.b1.ravel(),
                          self.w2.ravel(), self.b2.ravel(),
                          self.w3.ravel(), self.b3.ravel(),])
                         # self.w4.ravel(), self.b4.ravel()

    def set_params(self, params):
        input_dim, hidden_dim, hidden_dim_2, output_dim = self.input_dim, self.hidden_dim, self.hidden_dim_2, self.output_dim
        end_w1 = input_dim * hidden_dim
        end_b1 = end_w1 + hidden_dim
        end_w2 = end_b1 + hidden_dim*hidden_dim_2
        end_b2 = end_w2 + hidden_dim_2
        end_w3 = end_b2 + hidden_dim_2*output_dim
        end_b3 = end_w3 + output_dim
        # end_w4 = end_b3 + hidden_dim_2 * output_dim
        # end_b4 = end_w4 + output_dim

        self.w1 = params[:end_w1].reshape(input_dim, hidden_dim)
        self.b1 = params[end_w1:end_b1].reshape(1, hidden_dim)
        self.w2 = params[end_b1:end_w2].reshape(hidden_dim, hidden_dim_2)
        self.b2 = params[end_w2:end_b2].reshape(1, hidden_dim_2)
        self.w3 = params[end_b2:end_w3].reshape(hidden_dim_2, output_dim)
        self.b3 = params[end_w3:end_b3].reshape(1, output_dim)
        # self.w4 = params[end_b3:end_w4].reshape(hidden_dim_2, output_dim)
        # self.b4 = params[end_w4:end_b4].reshape(1, output_dim)

import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Numerical stability
    return exp_x / np.sum(exp_x)

class SGD:
    def __init__(self, lr=0.1, momentum=0.9, nesterov=False, weight_decay=0.0005):
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.velocity = None

    def step(self, gradients, params):
        if self.velocity is None:
            self.velocity = np.zeros_like(params)

        # Apply weight decay
        gradients += self.weight_decay * params

        if self.nesterov:
            prev_velocity = self.velocity
            self.velocity = self.momentum * self.velocity - self.lr * gradients
            params += -self.momentum * prev_velocity + (1 + self.momentum) * self.velocity
        else:
            self.velocity = self.momentum * self.velocity - self.lr * gradients
            params += self.velocity

        return params

def compute_loss_mse(y_true, y_pred, model, weight_decay=0.0005):
    mse_loss = np.mean((y_true - y_pred) ** 2)
    l2_loss = 0.5 * weight_decay * np.sum(model.get_params() ** 2)
    return mse_loss + l2_loss

def compute_loss_mse_class(y_true, y_pred, model, weight_decay=0.0005):
    """
    MSE loss for classification task.
    y_true: List of probabilities for each class (softmaxed)
    y_pred: Predicted values for each class (raw logits)
    """
    # Apply softmax to predictions
    #y_pred = softmax(y_pred)
    
    mse_loss = np.mean((np.array(y_true).reshape(1, -1) - np.array(y_pred).reshape(1, -1)) ** 2)
    l2_loss = 0.5 * weight_decay * np.sum(model.get_params() ** 2)
    return mse_loss + l2_loss

def compute_loss_xentropy(y_true, y_pred, model, weight_decay=0.0005):
    """
    Cross-Entropy loss for classification task.
    y_true: List of probabilities for each class (softmaxed)
    y_pred: Predicted values for each class (raw logits)
    """
    y_pred = softmax(y_pred)
    y_true = softmax(y_true)
    xentropy_loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / len(y_true)
    l2_loss = 0.5 * weight_decay * np.sum(model.get_params() ** 2)
    return xentropy_loss + l2_loss

def compute_gradients(model, X, y, weight_decay=0.0005, loss_type="xentropy"):
    epsilon = 1e-5
    original_params = model.get_params()
    n_params = len(original_params)
    gradients = np.zeros(n_params)
    
    if loss_type == "mse":
        loss_fn = compute_loss_mse
    elif loss_type == "mse_class":
        loss_fn = compute_loss_mse_class
    elif loss_type == "xentropy":
        loss_fn = compute_loss_xentropy
    
    for i in range(n_params):
        perturb = np.zeros(n_params)
        perturb[i] = epsilon
        model.set_params(original_params + perturb)
        loss1 = loss_fn(y, model.predict(X), model, weight_decay)
        model.set_params(original_params - perturb)
        loss2 = loss_fn(y, model.predict(X), model, weight_decay)
        gradients[i] = (loss1 - loss2) / (2 * epsilon)
    model.set_params(original_params)
    return gradients



def stochastic_train(
    model, X, y, batch_size=32, tolerance=2e-2, max_iterations=10000, lr=0.1, momentum=0.9, 
    weight_decay=0.0005, decay_rate=0.001, loss_type="xentropy"
):
    optimizer = SGD(lr=lr, momentum=momentum, weight_decay=weight_decay)
    n_samples = X.shape[0]
    iteration = 0
    loss = float('inf')

    while loss > tolerance and iteration < max_iterations:
        # Learning rate schedule
        optimizer.lr = lr / (1 + decay_rate * iteration)
        
        # Stochastic feature selection
        batch_indices = np.random.choice(n_samples, batch_size, replace=False)
        X_batch = X[batch_indices]
        y_batch = y[batch_indices]

        # Compute gradients
        gradients = compute_gradients(model, X_batch, y_batch, weight_decay)
        params = model.get_params()

        # Update parameters
        new_params = optimizer.step(gradients, params)
        model.set_params(new_params)

        y_predict = model.predict(X)
        loss_fn = compute_loss_mse if loss_type == "mse" else compute_loss_xentropy
        loss = loss_fn(y, y_predict, model, weight_decay)
        
        if iteration % 100 == 0 :
            print(f"Iteration {iteration}, Loss: {loss:.4f}, Learning Rate: {optimizer.lr:.6f}, rmse: {np.sqrt(mean_squared_error(y, y_predict))}, r2: {r2_score(y, y_predict)}")
        
        iteration += 1

    if loss <= tolerance:
        print("Training converged.")
    else:
        print("Reached maximum iterations without converging.")



class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = None
        self.v = None
        self.t = 0

    def step(self, gradients, params):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1

        # Apply weight decay
        if self.weight_decay > 0:
            gradients += self.weight_decay * params

        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(gradients)

        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1**self.t)
        # Compute bias-corrected second raw moment estimate
        v_hat = self.v / (1 - self.beta2**self.t)

        # Update parameters
        params -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return params



def adam_train(
    model, X, y, batch_size=32, tolerance_r2= 0.85, tolerance_rmse = 0.00001, max_iterations=10000, 
    lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0, loss_type= "mse_class"
):
    optimizer = Adam(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, weight_decay=weight_decay)
    n_samples = X.shape[0]
    iteration = 0
    loss = float('inf')
    r2 = 0
    rmse = float('inf')
    
    while (r2 < tolerance_r2 or rmse > tolerance_rmse) and iteration < max_iterations :
        # Stochastic batch selection
        batch_indices = np.random.choice(n_samples, batch_size, replace=False)
        X_batch = X[batch_indices]
        y_batch = y[batch_indices]
        
        try:
            # Compute gradients
            gradients = compute_gradients(model, X_batch, y_batch, weight_decay, loss_type= loss_type)
            params = model.get_params()

            # Update parameters
            new_params = optimizer.step(gradients, params)
            model.set_params(new_params)

            y_predict = model.predict(X)
            r2 = r2_score(y, y_predict)
            rmse = np.sqrt(mean_squared_error(y, y_predict))
        
        except Exception as e:
            print("y shape: ", y.shape, "y_pred shape: ", y_predict.shape)
            print(e)
        if iteration % 100 == 0 :
            print(f"Iteration {iteration}, Learning Rate: {optimizer.lr:.6f}, rmse: {np.sqrt(mean_squared_error(y, y_predict))}, r2: {r2_score(y, y_predict)}")

        iteration += 1

    if  r2 < tolerance_r2 and rmse > tolerance_rmse:
        print("Training converged.")
    else:
        print("Reached maximum iterations without converging.")

    return model

from multiprocessing import Pool

def compute_gradients_batch(args):
    """Helper function to compute gradients for a single batch."""
    model, X_batch, y_batch, weight_decay = args
    return compute_gradients(model, X_batch, y_batch, weight_decay)

def adam_train_kfold(
    model, X, y, batch_size=32, N=5, tolerance=1e-4, max_iterations=3000,
    lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0, n_cores = 4, loss_type = "mse"
):
    optimizer = Adam(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, weight_decay=weight_decay)
    n_samples = X.shape[0]
    N = batch_size // n_cores
    iteration = 0
    loss = float('inf')

    while loss > tolerance and iteration < max_iterations:
        # Generate `batch_size * N` random samples and split into N batches
        sample_indices = np.random.choice(n_samples, batch_size, replace=False)
        batches_per_core = np.array_split(sample_indices, n_cores)

        batch_args = [(model, X[batch_indices], y[batch_indices], weight_decay) for batch_indices in batches_per_core]

        with Pool(n_cores) as pool:
            # Parallel computation of gradients
            core_gradients = pool.map(compute_gradients_batch, batch_args)

        # Accumulate gradients from all cores
        gradients_accumulator = core_gradients[0]  # Initialize with first core's gradients
        for core_gradient in core_gradients[1:]:
            gradients_accumulator = [g_acc + g for g_acc, g in zip(gradients_accumulator, core_gradient)]

        # Average gradients across all cores
        gradients = [g / n_cores for g in gradients_accumulator]
        params = model.get_params()

        # Update parameters
        new_params = optimizer.step(gradients, params)
        model.set_params(new_params)

        # Evaluate the model on the full dataset after N batches
        y_predict = model.predict(X)
        loss_fn = compute_loss_mse if loss_type == "mse" else compute_loss_xentropy
        loss = loss_fn(y, y_predict, model, weight_decay)

        if iteration % 100 == 0 or loss <= tolerance:
            print(f"Iteration {iteration}, Loss: {loss:.4f}, Learning Rate: {optimizer.lr:.6f}, RMSE: {np.sqrt(mean_squared_error(y, y_predict))}, R2: {r2_score(y, y_predict)}")

        iteration += 1

    if loss <= tolerance:
        print("Training converged.")
    else:
        print("Reached maximum iterations without converging.")

    return model
