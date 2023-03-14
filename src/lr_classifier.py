import os

# machine learning tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# classificatio models
from sklearn.linear_model import LogisticRegression

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
    ap.add_argument("-t", "--tol",
        help="Tolerance for stopping criteria.", type = int, default=0.1)
    ap.add_argument("-s", "--solver",
        help="Algorithm to use in the optimization problem. Default is lbfgs. See scikit-learn documentatio for more info.", type = str, default = "saga")
    ap.add_argument("-p", "--penalty", help = "Specify the norm of the penalty.", type = str, default = "none")
    args = ap.parse_args()

    return args

def LR(tol_arg, solver_arg, penalty_arg):
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

    clf = LogisticRegression(penalty=penalty_arg,
                            tol=int(tol_arg),
                            verbose=True,
                            solver=solver_arg,
                            multi_class="multinomial").fit(X_train_dataset, y_train)

    y_pred = clf.predict(X_test_dataset)

    report = classification_report(y_test, 
                                y_pred, 
                                target_names=labels)
    print(report)

    # save classification report in out folder
    with open(os.path.join(os.getcwd(), "..", "out", "LR_report.txt"), "w") as f:
        f.write(report)

def main():
    args = input_parser()
    LR(args.tol, args.solver, args.penalty)

main()