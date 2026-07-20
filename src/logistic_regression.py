import numpy as np

def sigmoid(z):
    
    return 1 / (1 + np.exp(-z))

def compute_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    return loss

def compute_gradients(X, y, y_pred):
    m = X.shape[0]

    dw = (1 / m) * np.dot(X.T, (y_pred - y))
    db = (1 / m) * np.sum(y_pred - y)

    return dw, db

def gradient_descent(X, y, learning_rate = 0.01, iterations = 1000):
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0.0
    loss_history = []

    for _ in range(iterations):

        linear = np.dot(X, weights) + bias
        y_pred = sigmoid(linear)
        loss = compute_loss(y, y_pred)

        loss_history.append(loss)
        dw, db = compute_gradients(X, y, y_pred)
        weights -= learning_rate * dw
        bias -= learning_rate * db

    return weights, bias, loss_history

def predict_proba(X, weights, bias):
    
    linear = np.dot(X, weights) + bias
    
    return sigmoid(linear)

def predict(X, weights, bias, threshold = 0.5):
    
    probabilities = predict_proba(X, weights, bias)
    
    return (probabilities >= threshold).astype(int)

def accuracy(y_true, y_pred):
    
    return np.mean(y_true == y_pred)
