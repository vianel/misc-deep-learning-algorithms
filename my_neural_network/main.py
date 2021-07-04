import numpy as np

import matplotlib.pyplot as plt
from sklearn.datasets import make_gaussian_quantiles



def sigmoid(x, derivate = False):
    if derivate:
        return np.exp(-x)/(( np.exp(-x) +1)**2)
    else:
        return 1 / (1 + np.exp(-x))

def relu(x, derivate = False):
    if derivate:
        x[x<=0] = 0
        x[x>0] = 1
        return x
    else:
        return np.maximum(0,x)

def sigmoid(x, derivate=False):
    if derivate:
        return np.exp(-x)/((np.exp(-x)+1)**2)
    else:
        return 1/(1+np.exp(-x))

def initialize_parameters_deep(layer_dims):
    #np.random.seed(1)
    parameters = {}
    L = len(layer_dims)
    for l in range(0, L-1):
        parameters['W' + str(l+1)] = (np.random.rand(layer_dims[l], layer_dims[l+1]) * 2) - 1
        parameters['b' + str(l+1)] = (np.random.rand(1, layer_dims[l+1]) * 2) - 1
    return parameters

def mse(y, y_hat, derivate=False):
    if derivate:
        return (y_hat - y)
    else:
        return np.mean((y_hat - y)**2)

def train(x_data, lr, params, training=True):
    params['A0'] = x_data

    params['Z1'] = np.matmul(params['A0'], params['W1']) + params['b1']
    params['A1'] = relu(params['Z1'])

    params['Z2'] = np.matmul(params['A1'], params['W2']) + params['b2']
    params['A2'] = relu(params['Z2'])

    params['Z3'] = np.matmul(params['A2'], params['W3']) + params['b3']
    params['A3'] = sigmoid(params['Z3'])

    output = params['A3']

    if training:
        # Backpropagation

        params['dZ3'] =  mse(y,output,True) * sigmoid(params['A3'],True)
        params['dW3'] = np.matmul(params['A2'].T,params['dZ3'])

        params['dZ2'] = np.matmul(params['dZ3'],params['W3'].T) * relu(params['A2'],True)
        params['dW2'] = np.matmul(params['A1'].T,params['dZ2'])

        params['dZ1'] = np.matmul(params['dZ2'],params['W2'].T) * relu(params['A1'],True)
        params['dW1'] = np.matmul(params['A0'].T,params['dZ1'])


        ## Gradinet Descent:

        params['W3'] = params['W3'] - params['dW3'] * lr
        params['b3'] = params['b3'] - (np.mean(params['dZ3'],axis=0, keepdims=True)) * lr

        params['W2'] = params['W2'] - params['dW2'] * lr
        params['b2'] = params['b2'] - (np.mean(params['dZ2'],axis=0, keepdims=True)) * lr

        params['W1'] = params['W1'] -params['dW1'] * lr
        params['b1'] = params['b1'] - (np.mean(params['dZ1'],axis=0, keepdims=True)) * lr

    return output


if __name__ == "__main__":
    N = 1000
    gaussian_quantiles = make_gaussian_quantiles(mean=None,
                                                cov=0.1,
                                                n_samples=N,
                                                n_features=2,
                                                n_classes=2,
                                                shuffle=True,
                                                random_state=None)

    x, y = gaussian_quantiles
    y = y[:, np.newaxis]
    plt.scatter(x[:, 0], x[:, 1], c=y[:, 0], s=40, cmap=plt.cm.Spectral)
    plt.show()

    layers_dimemsions = [2, 4, 8, 1]
    params = initialize_parameters_deep(layers_dimemsions)
    print(params)

    print(params['W1'].shape)

    print(np.matmul(x, params['W1']).shape)

    errors = []

    print("-"*32)

    for _ in range(60000):
        output = train(x, 0.00001, params)
        if _ % 25 == 0:
            print(mse(y, output))
            errors.append(mse(y, output))

    plt.plot(errors)
    plt.show()

    data_test = (np.random.rand(1000, 2)* 2) - 1
    y_test = train(data_test, 0.001, params, False)

    y_test = np.where(y_test >= 0.5, 1, 0)
    print(y_test)

    plt.scatter(data_test[:, 0], data_test[:, 1], c=y_test[:, 0], s=40, cmap=plt.cm.Spectral)
    plt.show()
