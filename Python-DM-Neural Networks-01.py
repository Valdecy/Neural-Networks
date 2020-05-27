############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Data Mining
# Lesson: Neural Networks

# Citation: 
# PEREIRA, V. (2017). Project: Neural Networks, File: Python-DM-Neural Networks-01.py, GitHub repository: 
# <https://github.com/Valdecy/Neural Networks>

############################################################################

# Installing Required Libraries
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

# Function: Activation functions (Output & Hidden)
def sigmoid(a):
    return 1 / (1 + np.exp(-a)) # Sigmoid
def d_sigmoid(b):
    return b * (1.0 - b) # Derivative of Sigmoid
    
def tanh(a):
    return (2 / (1 + np.exp(-2*a))) -1 
def d_tanh(b):
    return 1.0 - b**2 

def identity(a):
    return a 
def d_identity(b):
    return (b*b + 1)/(b*b + 1)

def relu(a):
    return a * (a > 0)
def d_relu(b):
    return 1.0 * (b > 0)

def gaussian(a):
    return np.exp(-a**2)
def d_gaussian(b):
    return -2*b*np.exp(-b**2)

# Function: Activation function (Output)
def softmax(a):
    e = np.exp(a - np.max(a))  # prevent overflow
    if e.ndim == 1:
        return e / np.sum(e, axis = 0)
    else:  
        return e / np.array([np.sum(e, axis = 1)]).T

# Function: Confusion Matrix Plot
def plot_confusion_matrix(cm, classes, title = 'Confusion Matrix', cmap = plt.cm.Blues):
    
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment = "center", color = "white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

# Function: Binary Prediction
def predict_bin_nn(Xdata, weigths = ()):  
    
    architecture_width = len(weigths[0])
    activation         = weigths[1]
    output_activation  = weigths[2]
    
    input_layers      = [None] * architecture_width
    z_layers          = [None] * architecture_width
    activation_layers = [None] * architecture_width
    
    if (activation   == 'sigmoid'):
        act_function = sigmoid
    elif (activation == 'tanh'):
        act_function = tanh
    elif (activation == 'identity'):
        act_function = identity
    elif (activation == 'relu'):
        act_function = relu
    elif (activation == 'gaussian'):
        act_function = gaussian
    
    if (output_activation   == 'sigmoid'):
        out_act_function    = sigmoid
    elif (output_activation == 'tanh'):
        out_act_function    = tanh
    elif (output_activation == 'identity'):
        out_act_function    = identity
    elif (output_activation == 'relu'):
        out_act_function    = relu
    elif (output_activation == 'gaussian'):
        out_act_function    = gaussian
    elif (output_activation == 'softmax'):
        out_act_function    = softmax
                           
    input_layers[0] = np.append(np.ones(shape = (Xdata.shape[0], 1)), Xdata, axis = 1)
    for i in range(0, architecture_width):
        z_layers[i] = np.dot(input_layers[i], weigths[0][i])
        if (i == architecture_width - 1):
            activation_layers[architecture_width - 1] = out_act_function(z_layers[architecture_width - 1])
        elif (i < architecture_width - 1):
            activation_layers[i] = act_function(z_layers[i])
        if (i < architecture_width - 1):
            input_layers[i + 1] = np.append(np.ones(shape = (activation_layers[i].shape[0], 1)), activation_layers[i], axis = 1)    
    return np.where(activation_layers[architecture_width - 1] >= 0.5, 1.0, 0.0)

# Function: Linear Prediction
def predict_nn(Xdata, weigths = ()):
    
    architecture_width = len(weigths[0])
    activation         = weigths[1]
    output_activation  = weigths[2]
    
    input_layers      = [None] * architecture_width
    z_layers          = [None] * architecture_width
    activation_layers = [None] * architecture_width
    
    if (activation   == 'sigmoid'):
        act_function = sigmoid
    elif (activation == 'tanh'):
        act_function = tanh
    elif (activation == 'identity'):
        act_function = identity
    elif (activation == 'relu'):
        act_function = relu
    elif (activation == 'gaussian'):
        act_function = gaussian
    
    if (output_activation   == 'sigmoid'):
        out_act_function    = sigmoid
    elif (output_activation == 'tanh'):
        out_act_function    = tanh
    elif (output_activation == 'identity'):
        out_act_function    = identity
    elif (output_activation == 'relu'):
        out_act_function    = relu
    elif (output_activation == 'gaussian'):
        out_act_function    = gaussian
    elif (output_activation == 'softmax'):
        out_act_function    = softmax
                           
    input_layers[0] = np.append(np.ones(shape = (Xdata.shape[0], 1)), Xdata, axis = 1)
    for i in range(0, architecture_width):
        z_layers[i] = np.dot(input_layers[i], weigths[0][i])
        if (i == architecture_width - 1):
            activation_layers[architecture_width - 1] = out_act_function(z_layers[architecture_width - 1])
        elif (i < architecture_width - 1):
            activation_layers[i] = act_function(z_layers[i])
        if (i < architecture_width - 1):
            input_layers[i + 1] = np.append(np.ones(shape = (activation_layers[i].shape[0], 1)), activation_layers[i], axis = 1)    
    return activation_layers[architecture_width - 1]
    
# Function: nn - Neural Networks
def nn (Xdata, ydata, layers = [0], learning_rate = 0.01, activation = 'sigmoid', output_activation = 'sigmoid', loss = 'bin_cross_entropy', epochs = 5000):
    
    ################ Part 1 - Network Architecture #############################
    if (layers == [0]):
        activation = 'none'
        
    epochs = int(epochs)
    count = int(1)
    hidden_layers = np.array(layers[0])
    for i in range(0, len(layers)):
        if (i != 0):
            hidden_layers = np.append(hidden_layers, layers[i])
            count = count + 1
            
    hidden_layers = hidden_layers.reshape(1, count) # number of neurons in each hidden layer; hidden_layers.shape # (i,j); hidden_layers.size # number of elements; hidden_layers.shape[0] # rows; hidden_layers.shape[1] # columns or number of hidden layers
    number_of_hidden_layers = int(hidden_layers.shape[1]) # number of columns: number of hidden layers
    if (hidden_layers[0, 0] == 0):
        number_of_hidden_layers = int(0)
        
    architecture_width  = int(number_of_hidden_layers + 1) # architecture_width: number of set of weights
    architecture_length = np.array(Xdata.shape[1]) # architecture_length: number of weigths for each set
    for i in range(0, hidden_layers.shape[1]):
        if (hidden_layers[0, i] != 0):
            architecture_length = np.append(architecture_length, hidden_layers[0, i])
      
    if (output_activation == 'softmax'):
        out = 2
    else:
        out = 1
        
    architecture_length = np.append(architecture_length, out)
    architecture_length = architecture_length.reshape((1, architecture_length.shape[0]))
    
    # Activation functions
    if (activation     == 'sigmoid'):
        act_function   = sigmoid
        d_act_function = d_sigmoid
    elif (activation   == 'tanh'):
        act_function   = tanh
        d_act_function = d_tanh
    elif (activation   == 'identity'):
        act_function   = identity
        d_act_function = d_identity
    elif (activation   == 'relu'):
        act_function   = relu
        d_act_function = d_relu
    elif (activation   == 'gaussian'):
        act_function   = gaussian
        d_act_function = d_gaussian
    
    if (output_activation   == 'sigmoid'):
        out_act_function    = sigmoid
        d_out_act_function  = d_sigmoid
    elif (output_activation == 'tanh'):
        out_act_function    = tanh
        d_out_act_function  = d_tanh
    elif (output_activation == 'identity'):
        out_act_function    = identity
        d_out_act_function  = d_identity
    elif (output_activation == 'relu'):
        out_act_function    = relu
        d_out_act_function  = d_relu
    elif (output_activation == 'gaussian'):
        out_act_function    = gaussian
        d_out_act_function  = d_gaussian
    elif (output_activation == 'softmax'):
        out_act_function    = softmax
        
    # Feedfoward Lists 
    input_layers      = [None] * architecture_width
    z_layers          = [None] * architecture_width
    activation_layers = [None] * architecture_width
    error_layers      = [None] * architecture_width
    list_of_weigths   = [None] * architecture_width
                        
    # Backpropagation Lists 
    delta_layers      = [None] * architecture_width
    list_of_nablas    = [None] * architecture_width
                        
    # Validation Arrays
    loss_graph        = np.ones(shape = (epochs, 1))
    if (loss == 'bin_cross_entropy'):
        acc_graph         = np.ones(shape = (epochs, 1))
                        
    # Initial Weigths
    # np.random.seed(256)
    for i in range(0, architecture_width):
        r = int(architecture_length[0, i])
        c = int(architecture_length[0, i + 1])
        list_of_weigths[i] = np.random.rand(r + 1, c)/10

    # Algorithm
    for j in range(0, epochs):
        
        ##################### Part 2 - Feedforward #################################
        
        # Feedforward                            
        input_layers[0] = np.append(np.ones(shape = (Xdata.shape[0], 1)), Xdata, axis = 1)
        for i in range(0, architecture_width):
            z_layers[i] = np.dot(input_layers[i], list_of_weigths[i])
            if (i == architecture_width - 1):
                activation_layers[architecture_width - 1] = out_act_function(z_layers[architecture_width - 1])
            elif (i < architecture_width - 1):
                activation_layers[i] = act_function(z_layers[i])
            if (i < architecture_width - 1):
                input_layers[i + 1] = np.append(np.ones(shape = (activation_layers[i].shape[0], 1)), activation_layers[i], axis = 1)
        
        # Output Error
        error_layers[architecture_width - 1] = ydata - activation_layers[architecture_width - 1]
        
        # Loss Function
        if (loss == 'bin_cross_entropy'):
            J_matrix = -ydata*np.log10(activation_layers[architecture_width - 1]) - (1 - ydata)*np.log10(1 - activation_layers[architecture_width - 1])
            cost_J = np.sum(J_matrix)/J_matrix.shape[0]
        elif(loss == 'mse'):
            J_matrix = (ydata - activation_layers[architecture_width - 1])**2
            cost_J = np.sum(J_matrix)/2
              
        # Confusion Matrix
        if (loss == 'bin_cross_entropy'):
            y_pred = np.where(activation_layers[architecture_width - 1] >= 0.5, 1.0, 0.0)
            cm = confusion_matrix(ydata, y_pred) # from sklearn.metrics import confusion_matrix
            acc = (cm[0,0] + cm[1,1])/y_pred.shape[0]
            loss_graph[j, 0] = cost_J
            acc_graph[j, 0]  = acc
        
        if (loss == 'bin_cross_entropy'):         
            print('epochs: ' , j + 1 , ' loss function: ' , cost_J , ' acc: ' , acc)
        else:
            print('epochs: ' , j + 1 , ' loss function: ' , cost_J)
        
        ################### Part 3 - Backpropagation ###############################
    
        # Backpropagation   
        delta_layers[architecture_width - 1] = np.multiply(error_layers[architecture_width - 1], d_out_act_function(activation_layers[architecture_width - 1])) # Hadamart Multiplication
        
        # Error Layers
        for i in range(architecture_width - 1, 0, -1):
            error_layers[i - 1] = np.dot(delta_layers[i], np.delete(list_of_weigths[i], (0), axis = 0).T) # To delete the first row, do this: np.delete(array, (0), axis = 0); # To delete the third column, do this:  np.delete(array,(2), axis = 1)
            delta_layers[i - 1] = np.multiply(error_layers[i - 1], d_act_function(activation_layers[i - 1])) # Hadamart Multiplication
        
        # Nablas (Gradient Weigths Update)
        for i in range(0, architecture_width):
            r = int(architecture_length[0, i])
            c = int(architecture_length[0, i + 1])
            list_of_nablas[i] = np.zeros(shape = (r + 1, c)) # Zeros
        
        list_of_nablas[architecture_width - 1] = np.dot(input_layers[architecture_width - 1].T, delta_layers[architecture_width - 1])
        
        for i in range(architecture_width - 1, 0, -1):
            list_of_nablas[i - 1] = np.dot(input_layers[i - 1].T, delta_layers[i - 1]) # Zeros
        
        # Weigths Update
        if ((j + 1) < epochs):
            for i in range(0, architecture_width):
                list_of_weigths[i] = list_of_weigths[i] + learning_rate*list_of_nablas[i]
        
    # CM Plot
    if (loss == 'bin_cross_entropy'):
        plt.figure()
        plot_confusion_matrix(cm, classes = np.unique(ydata), title = 'Confusion Matrix', cmap = plt.cm.Blues)
        plt.show()
    
    # Loss and acc Plots
    if (loss == 'bin_cross_entropy'):
        plt.figure()
        plt.plot(loss_graph, label = 'loss')
        plt.plot(acc_graph,  label = 'acc')
        plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
        plt.show()
    else:
        plt.figure()
        plt.plot(loss_graph, label = 'loss')
        plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
        plt.show()
        
    return list_of_weigths, activation, output_activation

        
    ############### End of Function ##############
    
######################## Part 4 - Usage ####################################

# Binary Outuput
dataset = pd.read_csv('Python-DM-Neural Networks-02.csv', sep = ';')
# dataset from:
# FAVERO, L. P.; BELFIORE, P.; SILVA, F. L.; CHAN, B. (2009). Analise de Dados: Modelagem Multivariada para Tomada de Decisoes.CAMPUS.

dataset = dataset.replace(",", ".", regex = True)

X_bin = dataset.iloc[:, 2:7] # dataset needs to be scaled
X_bin = preprocessing.scale(X_bin) # from sklearn import preprocessing
X_bin = X_bin.reshape((X_bin.shape[0], X_bin.shape[1]))
y_bin = dataset.iloc[:, 1] # [:, 0] = first column       
y_bin = y_bin.values.reshape((y_bin.shape[0], 1))

model_bin = nn(X_bin, y_bin, layers = [4,4], learning_rate = 0.1, activation = 'sigmoid', output_activation = 'sigmoid', loss = 'bin_cross_entropy', epochs = 5000)

prediction_bin = predict_bin_nn(X_bin, weigths = model_bin)
comparison_bin = np.append(prediction_bin, y_bin, axis = 1)

# Linear Outuput
dataset = pd.read_csv('Python-DM-Neural Networks-03.csv', sep = ';')
dataset = dataset.replace(",", ".", regex = True)
# dataset from:
# PEREIRA, V. (2017). Project: Neural Networks, GitHub repository: <https://github.com/Valdecy/Neural Networks>

X_lin = dataset.iloc[:, 1:3] # dataset already scaled
# X_lin = preprocessing.scale(X_lin) # from sklearn import preprocessing
# X_lin = X_lin.reshape((X_lin.shape[0], X_lin.shape[1]))
y_lin = dataset.iloc[:, 0]       
y_lin = y_lin.values.reshape((y_lin.shape[0], 1))

model = nn(X_lin, y_lin, layers = [0], learning_rate = 0.01, output_activation = 'identity', loss = 'mse', epochs = 2000)

prediction = predict_nn(X_lin, weigths = model) 
comparison = np.append(prediction, y_lin, axis = 1)

########################## End of Code #####################################
