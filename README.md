# Neural Network Function

Pure python Neural Network Function for Binary or Linear Output. The function returns: 1) the final weigths, 2) the hidden layers activation function and 3) the output activation function

* Trainning: Full Batch.
* Activation Functions: "sigmoid", "tanh",  "identity", "relu" or "gaussian".
* Output Activation Functions: "sigmoid", "tanh",  "identity", "relu", or "gaussian".
* Epochs: The total number of iterations.
* Layers: List with the number of hidden layers and quantity of neurons. Ex: layers[0] = Perceptron; layers[1, 7] = Two hidden layers, the first one with 1 neuron and the second one with 7 neurons.
* Loss Function: "bin_cross_entropy" (Binary Output) or "mse" (Linear Output).
* For the Binary Output case a confusion matrix table and a loss function graph with the accuracy (acc) for each iteration is provided.
* For the Linear Output case a loss function graph for each iteration is provided.
* Finnaly a prediction function for the Binary Output (predict_bin_nn) and for the Linear Output (predict_nn) is also included.
