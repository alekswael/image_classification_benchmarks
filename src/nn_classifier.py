import os

# machine learning tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# classificatio models
from sklearn.neural_network import MLPClassifier

# data loader
from tensorflow.keras.datasets import cifar10

# import numpy and os
import numpy as np
import cv2

# import argparse
import argparse

def input_parser():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-hls", "--hidden_layer_sizes",
        help="The ith element represents the number of neurons in the ith hidden layer.", type = str, default="64,10")
    ap.add_argument("-i", "--max_iter",
        help="Maximum number of iterations.", type = int, default = 20)
    ap.add_argument("-l", "--learning_rate", help = "Learning rate schedule for weight updates", type = str, default = "adaptive")
    ap.add_argument("-s", "--early_stopping", help = "Whether to use early stopping to terminate training when validation score is not improving.", type = bool, default = True)
    args = ap.parse_args()

    return args

def NN(hidden_layer_sizes_arg, max_iter_arg, learning_rate_arg, early_stopping_arg):

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    labels = ["airplane", 
            "automobile", 
            "bird", 
            "cat", 
            "deer", 
            "dog", 
            "frog", 
            "horse", 
            "ship", 
            "truck"]

    X_train_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train])
    X_test_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test])

    # Just standardized?
    X_train_scaled = (X_train_grey)/255.0
    X_test_scaled = (X_test_grey)/255.0

    nsamples, nx, ny = X_train_scaled.shape
    X_train_dataset = X_train_scaled.reshape((nsamples,nx*ny))

    nsamples, nx, ny = X_test_scaled.shape
    X_test_dataset = X_test_scaled.reshape((nsamples,nx*ny))

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
    print(report)

    # save classification report in out folder
    with open(os.path.join(os.getcwd(), "out", "NN_report.txt"), "w") as f:
        f.write(report)

def main():
    args = input_parser()
    NN(args.hidden_layer_sizes, args.max_iter, args.learning_rate, args.early_stopping)

main()