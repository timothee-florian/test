# Identification of physical particles from a dataset

The aim of the code provided is to create a model based on a train set of physical particles and their parameters in order to differentiate Higgs Boson in a test set.

## Sections

### Helper for the project

Name : proj1_helpers.py

This section help with the basic work for exploiting the data (reading the data, writing new data,...).

#### Methods

- load_csv_data(data_path, sub_sample=False) : Load the data from a csv file and have 3 returns : one column containing 1 for an Higgs boson and -1 for the background, a numpy array of 30 columns with a value for each parameters and the indexes.
data_path : the path where the data is stored
sub_sample=True : create a sub sample of size one fifth of the original data, taking data every five rows

-predict_labels(weights, data) : One column of predictions is calculated (1 and -1).
weights : weight coefficients of the model chosen
data : numpy array containing values for differents parameters

-create_csv_submission(ids, y_pred, name) : Create a csv file containing the index and the prediction.
ids : index of the prediction
y_pred : prediction containing 1 or -1 values
name : name of the csv created

### Functions

Name : functions.py

This section gives all the building blocks that is needed for running the implementation methods along with the cross validation techniques. Furthermore, it calculates the loss of the prediction evaluated.

### Implementation

Name : implementations.py

This section contains all the important method used to give a good prediction while always returning the weights coefficients of the chosen model and the loss between the real value and the predicted one. These methods will help a lot to enhance the results that we have.

### Polynomial regression

Name : functions_for_polynomial.py

This section gives all the methods to make polynome regression using different techniques. Calculation of the degree with minimal error for each of the column and testing with different columns and differents degree to take the best combination.

### Plotting

Name : plot.py

This section gives a plot of the loss in function of either the number of degree for the polynomial regression or the hyperparameter lambda for the ridge regression.

## Notebook

### Combinatory

Name : combinatory.py



### Evaluation of the prediction

Name : evaluation_function.py

This section help to have a better understanding of the prediction not just the result in percent but as well for the quantity of 1 and -1 and the false positive and negative.
