import numpy as np
import h5py
from random import randint

#loading MNIST data
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] )
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32( MNIST_data['x_test'][:] )
y_test = np.int32( np.array( MNIST_data['y_test'][:,0] ) )
MNIST_data.close()

def sigmoid(z):
    return (1/(1+np.exp(-z)))

#derivative of tanh
def tanh_der(z):
    return (1-np.power(np.tanh(z),2))

#derivative of sigmoid function
def sigmoid_der(z):
    return sigmoid(z)*(1-sigmoid(z))

def softmax(z):
    array = np.exp(z - max(z))/(np.sum(np.exp(z - max(z))))
    return array

#calculating test data output
def test(X,C, b2, b1, W):
    X=np.reshape(X,(pixels,1))
    Z = np.matmul(W,X)+b1
    #H = sigmoid(Z)
    H = np.tanh(Z)
    U = np.matmul(C,H)+b2
    return softmax(U)

    
layer1_neurons = 100  #neurons in hidden layer
w_x = x_train.shape[1]
pixels = 28*28
num_epochs = 12
LR = 0.01

#parameters of neural network - initialization
W = np.random.randn(layer1_neurons,pixels)/np.sqrt(pixels)
b1 = np.random.randn(layer1_neurons,1)/np.sqrt(layer1_neurons)
C = np.random.randn(10,layer1_neurons)/np.sqrt(layer1_neurons)
b2 = np.random.randn(10,1)/np.sqrt(layer1_neurons)



for epoch in range(num_epochs):
    correct = 0
    for i in range(len(x_train)):
        #schedule for the learning rate
        if (epoch > 5):
            LR = 0.001
        if (epoch > 10):
            LR = 0.0001
        if (epoch > 15):
            LR = 0.00001
        
        #picking a random sample from training dataset
        n_random = randint(0,len(x_train)-1)
        y = y_train[n_random]
        X = x_train[n_random][:]
        
        X=np.reshape(X,(pixels,1))
        
        
        #forward step
        Z = np.matmul(W,X)+b1
        #H = sigmoid(Z)
        H =  np.tanh(Z) #activation vector
        U = np.matmul(C,H)+b2
        Fx = softmax(U) #activation vector
        pred = np.argmax(Fx) #output value with largest probability
        
        if (pred == y):
            correct +=1
        
        #creating an encoded vector for the actual digit value
        Y = np.zeros((10,1))
        Y[y,0]=1
        
        #backward propogation
        dU = Fx - Y
        dC = np.matmul(dU,H.T)
        d = np.matmul(C.T,dU)
        #d_sigdiff = np.multiply(d,sigmoid_der(Z))
        d_sigdiff = np.multiply(d,tanh_der(Z))
        
        
        #updating parameters
        C = C - LR*dC
        b2 = b2-LR*dU
        b1 = b1-LR*d_sigdiff
        W = W - LR*np.matmul(d_sigdiff,X.T)
        
    print('Epoch:'+str(epoch)+' Training data Accuracy: {}'.format((correct/np.float(len(x_train)))*100))
    

#Running model on test data        
correct = 0    
for i in range(len(x_test)):   
    y_calc = test(x_test[i],C, b2, b1, W) #returns result of softmax
    index = np.argmax(y_calc)
    if index == y_test[i]:
        correct+=1

accuracy = (correct/np.float(len(x_test)))*100     
print('\nTest data accuracy: {}'.format(accuracy))        
        
    