# Import the relevant components
from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)
import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
import sys
import os

import cntk as C
import cntk.tests.test_utils
cntk.tests.test_utils.set_device_from_pytest_env()
C.cntk_py.set_fixed_random_seed(1) # fix a random seed for CNTK components

#############################################################
# GLOBAL VARIABLES
#############################################################
# Number of output classes
num_output_classes = 5 # Number of room types (see Data/room_types.txt)

# K-fold cross validation
K = 10

# Model hyperparameters
num_hidden_layers = 2
learning_rate = 0.5
hidden_layers_dim = 10

# Parameters for the trainer
minibatch_size = 25
num_samples = 20000
num_minibatches_to_train = num_samples / minibatch_size

#############################################################
#############################################################

# Convert labels array to binary (target) arrays
def convertLabelsToBinaryArrays(labels, num_output_classes):

    # Convert labels (numbers) to binary arrays
    labelsBinary = []
    for i in range(labels.size):
        binaryArray = np.zeros(num_output_classes)
        binaryArray[labels[i]] = 1
        labelsBinary.append(binaryArray)

    labelsBinary = np.asarray(labelsBinary) # Change to numpy array and change data type
    return labelsBinary

# Randomly shuffle the input data
def shuffleData(inputData, labels):

    # Convert to numpy array and transpose
    labels = np.asarray([labels]).T

    # Join arrays
    data_appended = np.hstack((inputData, labels))   # Stack arrays horisontally (column wise)

    # Shuffle
    data_appended = np.random.permutation(data_appended)    # Randomly permute rows

    features_shuffled = np.delete(data_appended, -1, axis=1)    # Extract inputData (all but last column)
    labels_numbers_shuffled = np.asarray([data_appended[:,-1]]).T   # Extract labels (last column)
    return features_shuffled, labels_numbers_shuffled

def linear_layer(input_var, output_dim):
    input_dim = input_var.shape[0]

    weight = C.parameter(shape=(input_dim, output_dim))
    bias = C.parameter(shape=(output_dim))

    return bias + C.times(input_var, weight)



def dense_layer(input_var, output_dim, nonlinearity):
    l = linear_layer(input_var, output_dim)

    return nonlinearity(l)


# Define a multilayer feedforward classification model
def fully_connected_classifier_net(input_var, num_output_classes, hidden_layer_dim,
                                   num_hidden_layers, nonlinearity):

    h = dense_layer(input_var, hidden_layer_dim, nonlinearity)
    for i in range(1, num_hidden_layers):
        h = dense_layer(h, hidden_layer_dim, nonlinearity)

    return linear_layer(h, num_output_classes)

def create_model(features):
    with C.layers.default_options(init=C.layers.glorot_uniform(), activation=C.sigmoid):
        h = features
        for _ in range(num_hidden_layers):
            h = C.layers.Dense(hidden_layers_dim)(h)
        last_layer = C.layers.Dense(num_output_classes, activation = None)

        return last_layer(h)

# Define a utility function to compute the moving average sum.
# A more efficient implementation is possible with np.cumsum() function
def moving_average(a, w=10):
    if len(a) < w:
        return a[:]    # Need to send a copy of the array
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]


# Defines a utility that prints the training progress
def print_training_progress(trainer, mb, frequency, verbose=1):
    training_loss = "NA"
    eval_error = "NA"

    if mb%frequency == 0:
        training_loss = trainer.previous_minibatch_loss_average
        eval_error = trainer.previous_minibatch_evaluation_average
        if verbose:
            print ("Minibatch: {}, Train Loss: {}, Train Error: {}".format(mb, training_loss, eval_error))

    return mb, training_loss, eval_error

"""
def showGraphs(plotdata):

    plt.figure(1)
    plt.subplot(311)
    plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs. Training loss')
    plt.subplot(313)
    plt.plot(plotdata["batchsize"], plotdata["avgerror"], 'r--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Label Prediction Error')
    plt.title('Minibatch run vs. Label Prediction Error')
    plt.show()
"""

# Compute classificationRate and confusion matrix based on true and predicted labels
def computeMetrics(true_labels, predicted_labels):

    # Confusion matrix
    # Rows - actual, Columns - predicted
    confusionMatrix = np.zeros((num_output_classes, num_output_classes))
    correct_guesses = 0
    for i in range(len(true_labels)):
        confusionMatrix[true_labels[i]][predicted_labels[i]] += 1
        if(true_labels[i] == predicted_labels[i]):
            correct_guesses += 1

    classificationRate = correct_guesses/len(true_labels)

    return (classificationRate, confusionMatrix)

def defineModel(input_dim):
    input = C.input_variable(input_dim)
    modelLabel = C.input_variable(num_output_classes)

    # Create the fully connected classfier
    model = fully_connected_classifier_net(input, num_output_classes, hidden_layers_dim,
                                       num_hidden_layers, C.sigmoid)
    model = create_model(input)

    return (input, model, modelLabel)

def trainAndTestOneFold(model, modelLabel, features, labels, features_test, labels_test):
    input = model.arguments[0]

    # Training
    loss = C.cross_entropy_with_softmax(model, modelLabel)
    eval_error = C.classification_error(model, modelLabel)

    # Instantiate the trainer object to drive the model training
    lr_schedule = C.learning_parameter_schedule(learning_rate)
    learner = C.sgd(model.parameters, lr_schedule)
    trainer = C.Trainer(model, (loss, eval_error), [learner])

    # Run the trainer and perform model training
    training_progress_output_freq = 20

    plotdata = {"batchsize":[], "loss":[], "error":[]}
    for i in range(0, int(num_minibatches_to_train)):
        # Specify the input variables mapping in the model to actual minibatch data for training
        trainer.train_minibatch({input : features, modelLabel : labels})
        batchsize, loss, error = print_training_progress(trainer, i,
                                                         training_progress_output_freq, verbose=0)

        if not (loss == "NA" or error =="NA"):
            plotdata["batchsize"].append(batchsize)
            plotdata["loss"].append(loss)
            plotdata["error"].append(error)

    # Compute the moving average loss to smooth out the noise in SGD
    plotdata["avgloss"] = moving_average(plotdata["loss"])
    plotdata["avgerror"] = moving_average(plotdata["error"])

    # Graph data
    #showGraphs(plotdata)

    trainer.test_minibatch({input : features_test, modelLabel : labels_test})

    out = C.softmax(model)
    predicted_label_probs = out.eval({input : features_test})

    true_labels = [np.argmax(label) for label in labels_test]
    predicted_labels = [np.argmax(row) for row in predicted_label_probs]
    classificationRate, confusionMatrix = computeMetrics(true_labels, predicted_labels)

    print("Label    :", true_labels)
    print("Predicted:", predicted_labels)
    print("Precision: ", classificationRate)
    print("Confusion Matrix:\n", confusionMatrix)

    return (classificationRate, confusionMatrix)

#############################################################
# MAIN
#############################################################
if __name__ == "__main__":

    # Load data and labels for that dataset
    inputData_raw = np.load('Data/object_result.dat')
    labels_raw = np.load('Data/types_result.dat')

    # Shuffle data randomly
    inputData_raw, labels_raw = shuffleData(inputData_raw, labels_raw)

    # Define the data dimensions
    input_dim = inputData_raw.shape[1]
    num_examples = inputData_raw.shape[0]
    training_size = int(np.floor(num_examples * (1 - 1/K)))
    test_size = num_examples - training_size

    # Print shapes
    print("\n### DATA LOADED")
    print("Input Data and labels (numbers) shape: ", inputData_raw.shape, labels_raw.shape)
    print("Training set size: ", training_size, "Test size: ", test_size)

    # Process data
    inputData = np.float32(inputData_raw) # Cast inputData type to float32
    labels = convertLabelsToBinaryArrays(labels_raw, num_output_classes)
    labels = np.float32(labels)

    confusionMatrixSum = np.zeros((num_output_classes, num_output_classes))
    for fold in range(K):
        lower_bound = fold * test_size # Inclusive
        upper_bound = fold * test_size + test_size # Exclusive

        print("Lower bound (inclusive): ", lower_bound, "Upper bound (exclusive): ", upper_bound)

        features_test = inputData[lower_bound:upper_bound]
        labels_test = labels[lower_bound:upper_bound]
        features_training = np.delete(inputData, np.s_[lower_bound:upper_bound], axis=0)
        labels_training = np.delete(labels, np.s_[lower_bound:upper_bound], axis=0)

        print("Fold number: ", fold + 1)

        input, model, modelLabel = defineModel(input_dim)

        # Save the model for further use
        modelName = "FFNN_classificator_network"
        model.save(modelName + "." + str(fold) + "_fold" + ".dnn")
        model = C.ops.functions.Function.load(modelName + "." + str(fold) + "_fold" + ".dnn")

        classificationRate, confusionMatrix = trainAndTestOneFold(model, modelLabel,
                                            features_training, labels_training, features_test, labels_test)
        confusionMatrixSum += confusionMatrix
        print("\n\n")

    classificationRate = np.trace(confusionMatrixSum)/num_examples
    averageConfusionMatrix = confusionMatrixSum/num_examples
    print("Sum confusion matrix:\n", confusionMatrixSum)
    print("Average classification rate: ", classificationRate)

    precisionVector = np.zeros(num_output_classes)
    recallVector = np.zeros(num_output_classes)
    F1Vector = np.zeros(num_output_classes)

    for class_ in range(num_output_classes):
        if(np.sum(confusionMatrixSum[:,class_]) == 0):
            precisionVector[class_] = 1
        else:
            precisionVector[class_] = confusionMatrixSum[class_][class_]/np.sum(confusionMatrixSum[:,class_])

        if(np.sum(confusionMatrixSum[class_ , :]) == 0):
            recallVector[class_] = 1
        else:
            recallVector[class_] = confusionMatrixSum[class_][class_]/np.sum(confusionMatrixSum[class_ , :])

        F1Vector[class_] = 2 * precisionVector[class_] * recallVector[class_] / (precisionVector[class_] + recallVector[class_])

    # Plot confusion matrix
    cm = sns.heatmap(confusionMatrixSum/(np.sum(confusionMatrixSum, axis=1, keepdims=1)),
                    annot=True,
                    xticklabels=['Bathroom', 'Bedroom', 'Kitchen', 'Living room', 'Office'],
                    yticklabels=['Bathroom', 'Bedroom', 'Kitchen', 'Living room', 'Office'])
    sns.set(font_scale=1.4)
    plt.xlabel('Predicted room type')
    plt.ylabel('Actual room type')
    plt.title('Confusion Matrix for Room Classification using Neural Networks')
    plt.show()

    print("Precision vector: ", precisionVector)
    print("Recall vector: ", recallVector)
    print("F1 Score vector: ", F1Vector)
