# py packages
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

# Classification report tools
from sklearn.metrics import classification_report

# Classification models
from sklearn.linear_model import LogisticRegression

# Image processing tools
import cv2

# Data loader
from tensorflow.keras.datasets import cifar10

# Data management tools
import numpy as np


def input_parser(): # This is the function that parses the input arguments when run from the terminal.
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter) # This is the argument parser. I add the arguments below.
    ap.add_argument("-t",
                    "--tol",
                    help="Tolerance for stopping criteria.",
                    type = int, default=0.1) # This is the argument for the tolerance.
    ap.add_argument("-s",
                    "--solver",
                    help="Algorithm to use in the optimization problem. Default is lbfgs. See scikit-learn documentatio for more info.",
                    type = str,
                    default = "saga") # This is the argument for the solver
    ap.add_argument("-m",
                    "--max_iter",
                    help = "Maximum number of iterations taken for the solvers to converge.",
                    type = int,
                    default = 100) # This is the argument for the maximum number of iterations
    ap.add_argument("-p",
                    "--penalty",
                    help = "Specify the norm of the penalty.",
                    type = str,
                    default = "none") # This is the argument for the penalty
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

def LR(X_train_dataset, X_test_dataset, y_train, y_test, labels, tol_arg, solver_arg, max_iter_arg, penalty_arg): # This is the main function that runs the logistic regression model. 
    clf = LogisticRegression(penalty=penalty_arg, # Default is none.
                            tol=int(tol_arg), # Default is 0.1.
                            verbose=True,
                            solver=solver_arg, # Default is saga.
                            max_iter=max_iter_arg, # Default is 100.
                            multi_class="multinomial")
    
    clf.fit(X_train_dataset, y_train) # Fit the model to the training data.

    y_pred = clf.predict(X_test_dataset) # Predict the labels for the test data.

    report = classification_report(y_test, 
                                y_pred, 
                                target_names=labels) # Generate a classification report.
    
    print("Classification report: ") # Print the classification report.
    print(report)

    # Save classification report in out folder.
    with open(os.path.join(os.getcwd(), "out", "LR_report.txt"), "w") as f:
        f.write(report)

def main(): # This is the main function that runs the program.
    args = input_parser() # Parse the input arguments.
    X_train, y_train, X_test, y_test, labels = import_data() # Import the data.
    X_train_dataset, X_test_dataset = preprocess_data(X_train, X_test) # Preprocess the data.
    LR(X_train_dataset, X_test_dataset, y_train, y_test, labels, args.tol, args.solver, args.max_iter, args.penalty) # Run the logistic regression model.

if __name__ == "__main__":
    main()