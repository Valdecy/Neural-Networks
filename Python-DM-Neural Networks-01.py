############################################################################

# NN
# Prof. Valdecy Pereira
# e-mail: valdecy.pereira@gmail.com
# site: github.com/Valdecy

############################################################################

# Installing Required Libraries
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

# Function: Activation functions
def sigmoid(a):
    return 1 / (1 + np.exp(-a)) # Sigmoid
def d_sigmoid(b):
    return b * (1.0 - b) # Derivative of Sigmoid
def tanh(a):
    return np.tanh(a) # Tanh
def d_tanh(b):
    return 1.0 - b**2 # Derivative of Tanh

# Function: nn
def nn (X, y, layers = [3], learning_rate = 0.5, activation = 'sigmoid', epochs = 5000):
    
    ################ Part 1 - Network Architecture #############################
    
    epochs = int(epochs)
    count = int(1)
    hidden_layers = np.array(layers[0])
    for i in range(0, len(layers)):
        if (i != 0):
            hidden_layers = np.append(hidden_layers, layers[i])
            count = count + 1
            
    hidden_layers = hidden_layers.reshape(1, count) 
    number_of_hidden_layers = int(hidden_layers.shape[1]) 
    if (hidden_layers[0, 0] == 0):
        number_of_hidden_layers = int(0)
        
    architecture_width  = int(number_of_hidden_layers + 1) 
    architecture_length = np.array(X.shape[1]) 
    for i in range(0, hidden_layers.shape[1]):
        if (hidden_layers[0, i] != 0):
            architecture_length = np.append(architecture_length, hidden_layers[0, i])
            
    architecture_length = np.append(architecture_length, 1) 
    architecture_length = architecture_length.reshape((1, architecture_length.shape[0]))
    
    # Activation functions
    if (activation == 'sigmoid'):
        act_function = sigmoid
        d_act_function = d_sigmoid
    elif (activation == 'tanh'):
        act_function = tanh
        d_act_function = d_tanh
        
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
    acc_graph         = np.ones(shape = (epochs, 1))
                        
    # Initial Weigths
    for i in range(0, architecture_width):
        r = int(architecture_length[0, i])
        c = int(architecture_length[0, i + 1])
        list_of_weigths[i] = np.random.rand(r + 1, c)

    # Algorithm
    for j in range(0, epochs):
        
        ##################### Part 2 - Feedforward #################################
        
        # Feedforward                            
        input_layers[0] = np.append(np.ones(shape = (X.shape[0], 1)), X, axis = 1)
        for i in range(0, architecture_width):
            z_layers[i] = np.dot(input_layers[i], list_of_weigths[i])
            activation_layers[i] = act_function(z_layers[i])
            if (i < architecture_width - 1):
                input_layers[i + 1] = np.append(np.ones(shape = (activation_layers[i].shape[0], 1)), activation_layers[i], axis = 1)
        
        # Output Error
        error_layers[architecture_width - 1] = y - activation_layers[architecture_width - 1]
        
        # Loss Function
        if (activation == 'sigmoid'):
            J_matrix = np.multiply(-y, np.log10(activation_layers[architecture_width - 1])) - np.multiply((1 - y), np.log10(1 - activation_layers[architecture_width - 1]))
            cost_J = np.sum(J_matrix)/J_matrix.shape[0]
        elif (activation == 'tanh'):
            J_matrix = (y - activation_layers[architecture_width - 1])**2
            cost_J = np.sum(J_matrix)/2
              
        # Confusion Matrix
        y_pred = np.where(activation_layers[architecture_width - 1] >= 0.5, 1, 0)
        cm = confusion_matrix(y, y_pred) 
        acc = (cm[0,0] + cm[1,1])/y_pred.shape[0]
        loss_graph[j, 0] = cost_J
        acc_graph[j, 0]  = acc
        print('epochs: ' , j + 1 , ' cost function: ' , cost_J , ' acc: ' , acc)
        
        ################### Part 3 - Backpropagation ###############################
    
        # Backpropagation   
        delta_layers[architecture_width - 1] = np.multiply(error_layers[architecture_width - 1], d_act_function(activation_layers[architecture_width - 1]))
		
        # Error Layers
        for i in range(architecture_width - 1, 0, -1):
            error_layers[i - 1] = np.dot(delta_layers[i], np.delete(list_of_weigths[i], (0), axis = 0).T) 
            delta_layers[i - 1] = np.multiply(error_layers[i - 1], d_act_function(activation_layers[i - 1])) 
        
        # Nablas (Gradient Weigths Update)
        for i in range(0, architecture_width):
            r = int(architecture_length[0, i])
            c = int(architecture_length[0, i + 1])
            list_of_nablas[i] = np.zeros(shape = (r + 1, c))
        
        list_of_nablas[architecture_width - 1] = np.dot(input_layers[architecture_width - 1].T, delta_layers[architecture_width - 1])
        
        for i in range(architecture_width - 1, 0, -1):
            list_of_nablas[i - 1] = np.dot(input_layers[i - 1].T, delta_layers[i - 1]) 
        
        # Weigths Update
        for i in range(0, architecture_width):
            list_of_weigths[i] = list_of_weigths[i] + learning_rate*list_of_nablas[i]
    
    # CM
    print('Confusion Matrix:')
    for line in cm:
        print(*line)
           
    # Loss and acc Plots
    plt.figure()
    plt.plot(loss_graph)
    plt.plot(acc_graph)
    plt.show()
    
    # Final Weigths
    for i in range(0, architecture_width):
        print(list_of_weigths[i])
        
    ############### End of Function ##############
    
######################## Part 4 - Usage ####################################

# Importing Data
dataset = pd.read_csv('Python-DM-Neural Networks-02.csv', sep = ';')
dataset = dataset.replace(",", ".", regex = True)

X = dataset.iloc[:, 2:7].values
X = preprocessing.scale(X) 
X = X.reshape((X.shape[0], X.shape[1]))
y = dataset.iloc[:, 1].values
y = y.reshape((y.shape[0], 1))

nn(X, y, layers = [3, 2], learning_rate = 0.1, activation = 'sigmoid', epochs = 5000)

########################## End of Code #####################################




