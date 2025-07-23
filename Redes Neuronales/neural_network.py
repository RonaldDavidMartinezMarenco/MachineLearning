import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def main():
    # DATA PREPARE
    data = pd.read_csv('~/ml/Redes Neuronales/data/train.csv')
    data = np.array(data)
    m, n = data.shape
    np.random.shuffle(data) # shuffle before splitting into dev and training sets

    data_dev = data[0:1000].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:n]
    X_dev = X_dev / 255.

    data_train = data[1000:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n]
    X_train = X_train / 255.
    _,m_train = X_train.shape

    '''
    print(data)
    data = np.array(data)
    m,n = data.shape
    print(f"{m} x {n}")
    
    np.random.shuffle(data) # evitar sesgos por algun orden en mi dataset (baraja el dataset)
    
    data_dev = data[0:1000].T # se toman los primeros 1000 registros para validacion
    
    m2,n2 = data_dev.shape
    print(f"{m2} x {n2}")
    
    Y_dev = data_dev[0] 
    X_dev = data_dev[1:n] # desde 1 hasta 785 ya que intercambiamos filas por columnas
    
    data_train = data[1000:m].T # se toman desde 1000 hasta el resto de imagenes
    Y_train = data_train[0] # toma las etiquetas ya que esta quedan en el primer lugar
    X_train = data_train[1:n]# toma el resto de de data desde 1 hasta 785
    
    print(X_train)
    print(X_train[:, 0].shape)
    '''
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)
    
    # predictions
    test_prediction(5,W1,b1,W2,b2,X_train,Y_train)
    
    
def init_params():
    W1 = np.random.rand(10,784) - 0.5  # crea pesas entre 0 a 1 para 10 neuronlas hidden y 784 neuronas de entrada
    b1 = np.random.rand(10,1) - 0.5   # sesgos
    W2 = np.random.rand(10,10)  - 0.5 # crea pesas entre 0 a 1 para 10 neuronas hidden y 10 neuronas de entrada
    b2 = np.random.rand(10,1) - 0.5   # coloca el rango entre -0,5 y 0.5 para evitar ser aplanados por funciones de activacion, que aprendan patrones
    return W1,b1,W2,b2

def ReLU(Z):
    return np.maximum(Z,0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1)) # matriz de ceros de m(muestra de elementos) x 10 (9 elementos + 1)
    one_hot_Y [np.arange(Y.size),Y] = 1   # con 4 muestras sera 4 x 10 todas cero. donde mi matriz de etiquetas tenga un numero dicho numero sera el indice q se mapeara como 1
                                            # o sea si Y[0] es 2 entonces one_hot_Y = en su fila 0 se rellena con 1 el indice q pertenece al 2. y asi con el resto de fila
    one_hot_Y = one_hot_Y.T  # se transpone para poder compararlo con el output final restandole el numero correspondiente al indice
    return one_hot_Y
    
def ReLU_deriv(Z):
    return Z > 0  # True conviernte a 1 y False convierte a 0 . Lo cual es conveniente para ReLU
    
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2


def update_params(W1,b1,W2,b2,dW1,db1,dW2,db2,alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1,b1,W2,b2
    
def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2, X_train, Y_train):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.savefig("number.png")
    #plt.show()

if __name__ == "__main__":
    main()