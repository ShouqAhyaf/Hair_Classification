# HairClassifier: Hair Type Classification Using AI
## Table of Contents:
- Introduction
- Technologies
- Components Required
- Usage Instructions
- Model
- Code

## Introduction
This project demonstrates the classification of hair types using an AI-powered image recognition model. The trained model categorizes images into four distinct hair types: Straight, Wavy, Curly, and Kinky.

## Technologies
Python: Version 3.8 or higher

TensorFlow: Version 2.x

Jupyter Notebook: For running the provided `HairClassification.ipynb` file

## Components Required
`keras_model.h5`: Pre-trained TensorFlow model file

`labels.txt`: Contains hair type labels (Straight, Wavy, Curly, Kinky)

Test Images: Example images like `Test1.jpg`, `Test2.jpg`, and `Test3.jpg`

## Usage Instructions
1. Setup
Install required libraries using pip install tensorflow pillow.
Ensure that the following files are in your project directory:
`keras_model.h5`
`labels.txt`
Test images (`Test1.jpg`, `Test2.jpg`, `Test3.jpg`).

2. Run the Classifier
Open the `HairClassification.ipynb` notebook.
Follow the instructions in the notebook to load the model, preprocess the images, and classify hair types.

## Model
- Labels:
The hair types are classified into the following categories:

0: Straight

1: Wavy

2: Curly

3: Kinky

- Classification Results:

The output includes:

The predicted hair type.

A confidence score for each classification.

## Code
The complete implementation can be found in `HairClassification.ipynb`.

