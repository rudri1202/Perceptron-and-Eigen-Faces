ASSIGNMENT-3

PROBLEM-1:
data.txt and main.py:
In this problem first we have to generate the synthetic data randomly so using numpy we generate 1000 rows of 4 integers lying between -100 to 100.
we also generate random weights for our linear function to firstly train our model by computing the target variable.
and furthur splitting the dataset in 70-30 split we obtain our train and test data respectively.

train.py:
we define functions and using these functions we do the following tasks.
this file takes the input file path of the training data and separates the target variable.
then it normalizes the dataset and takes the data to implement our perceptron that is predicting the weights by iterating through the dataset and adjusting the values to fit our dataset returning the weights file.

test.py:
we define functions and using these functions we do the following tasks.
firstly we normalize our test dataset.
we load the weights file into our code and using those weights we predict the labels of our test dataset and mention them as the output.





PROBLEM-2:
In this statement firstly we use the dataset provided by sklearn which is the labelled faces in wild (LFW) dataset.
we load the dataset in our local variables and prepareit and gain information about our dataset and display one the sample images after size reduction.
then we use train_test_split function from sklearn library to split our dataset
agin using the sklearn library we scale the features of training and testing data.
using matplot lib we plot the graph.
using PCA from sklearn library we perform dimensionality reduction on our dataset and thus obtain eigen faces.
then displaying some of the eigen faces
then using the sklearn library again we use knn classifier to compute labels for the testing data
furthur to experiment with our data and analyse the model we plot different graphs and make the confusion matrix for the same.

