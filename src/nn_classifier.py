# py packages
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

# Classification report tools
from sklearn.metrics import classification_report

# Classification models
from sklearn.neural_network import MLPClassifier

# Image processing tools
import cv2

# Data loader
from tensorflow.keras.datasets import cifar10

# Data management tools
import numpy as np

def input_parser(): # This is the function that parses the input arguments when run from the terminal.
    ap = argparse.ArgumentParser() # This is the argument parser. I add the arguments below.
    ap.add_argument("-hls",
                    "--hidden_layer_sizes",
                    help="The ith element represents the number of neurons in the ith hidden layer. If a single layer, DO NOT put a comma. Specify values WITHOUT SPACES.",
                    type = str,
                    default="64,10")
    ap.add_argument("-i", "--max_iter",
                    help="Maximum number of iterations.",
                    type = int,
                    default = 70)
    ap.add_argument("-l",
                    "--learning_rate",
                    help = "Learning rate schedule for weight updates",
                    type = str,
                    default = "adaptive")
    ap.add_argument("-s",
                    "--early_stopping",
                    help = "Whether to use early stopping to terminate training when validation score is not improving.",
                    type = bool,
                    default = True)
    args = ap.parse_args() # Parse the args
    return args


def import_data(): # This is the function that imports the data.
    (X_train, y_train), (X_test, y_test) = cifar10.load_data() # Load the data from the cifar10 dataset.

    labels = ["airplane", 
            "automobile", 
            "bird", 
            "cat", 
            "deer", 
            "dog", 
            "frog", 
            "horse", 
            "ship", 
            "truck"] # These are the labels for the data so we can interpret the results.
    
    print("Data imported.")
    
    return X_train, y_train, X_test, y_test, labels


def preprocess_data(X_train, X_test): # This is the function that preprocesses the data.
    X_train_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train]) # Convert the images to greyscale.
    X_test_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test])

    X_train_scaled = (X_train_grey)/255.0 # Scale the data to be between 0 and 1.
    X_test_scaled = (X_test_grey)/255.0

    nsamples, nx, ny = X_train_scaled.shape # Reshape the data to be 2D.
    X_train_dataset = X_train_scaled.reshape((nsamples,nx*ny))

    nsamples, nx, ny = X_test_scaled.shape # Same for the test data.
    X_test_dataset = X_test_scaled.reshape((nsamples,nx*ny))
    
    print("Data preprocessed.")
    
    return X_train_dataset, X_test_dataset

def NN(X_train_dataset, X_test_dataset, y_train, y_test, labels, hidden_layer_sizes_arg, max_iter_arg, learning_rate_arg, early_stopping_arg):

    nn = MLPClassifier(hidden_layer_sizes=tuple([int(i) for i in hidden_layer_sizes_arg.split(",")]),
                            max_iter = max_iter_arg,
                            verbose = True,
                            learning_rate = learning_rate_arg,
                            early_stopping = early_stopping_arg)

    nn.fit(X_train_dataset, y_train)

    y_pred = nn.predict(X_test_dataset)

    report = classification_report(y_test, 
                                y_pred, 
                                target_names=labels)
    
    print("Classification report: ")
    print(report)

    # save classification report in out folder
    with open(os.path.join(os.getcwd(), "out", "NN_report.txt"), "w") as f:
        f.write(report)

def main():
    args = input_parser()
    X_train, y_train, X_test, y_test, labels = import_data()
    X_train_dataset, X_test_dataset = preprocess_data(X_train, X_test)
    NN(X_train_dataset, X_test_dataset, y_train, y_test, labels, args.hidden_layer_sizes, args.max_iter, args.learning_rate, args.early_stopping)

main()