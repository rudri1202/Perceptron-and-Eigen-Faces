# -*- coding: utf-8 -*-
"""eigenfaces.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zHcx9RE5WVjPLvQk3pwShShAz2Yi0FjT
"""

from time import time
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import loguniform

from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

faces = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
t_images, h, w = faces.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = faces.data
dimensions = X.shape[1]

# the label to predict is the id of the person
y = faces.target
target_names = faces.target_names
u_images = target_names.shape[0]


print("Total dataset size:")
print("total images: %d" % t_images)
print("dimensions: %d" % dimensions)
print("unique images: %d" % u_images)

print("Image from out dataset")
image = X[7].reshape((h,w))
plt.imshow(image, cmap = 'gray')
plt.axis('off')

plt.show

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA().fit(X_train)
plt.figure(figsize= (10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel ('Number of Components')
plt.ylabel( 'Cumulative Explained Variance Ratio')
plt.title('Explained Variance Ratio vs. Number of Components')
plt.grid (True)
plt.show()

pca_comp = 200

print("Extracting the top %d eigenfaces from %d faces" % (pca_comp, X_train.shape[0]))


pca = PCA(n_components=pca_comp, svd_solver="auto", whiten=True).fit(X_train)


eigenfaces = pca.components_.reshape((pca_comp, h, w))

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

def display_eigenfaces(eigenfaces,y):

  for i in np.unique(y):

    plt.figure(figsize = (4,4))
    plt.imshow(eigenfaces[i], cmap = 'gray')
    plt.title('Eigenface {}'.format(i + 1))
    plt.axis('off')
    plt.show()

  return


show_images = display_eigenfaces(eigenfaces,y)

def accuracy_knn(X_train_pca, X_test_pca, y_train, y_test):


  knn = KNeighborsClassifier(n_neighbors = 6)
  knn.fit(X_train_pca, y_train)
  y_pred = knn.predict(X_test_pca)
  accuracy = accuracy_score(y_test, y_pred)
  return accuracy,y_pred

acc, y_pred = accuracy_knn(X_train_pca, X_test_pca, y_train, y_test)
print("Accuracy:", acc)

"""# Experimenting with data and Visualizing dataset

"""

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, target_names=faces.target_names))

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Assuming y_test and y_pred are your true and predicted labels, respectively

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

comp = np.arange(20)

def dif_accuracy(X_train, comp, X_test, y_train, y_test):

  accuracies = []

  for i in range(comp.size):



    n_components = (comp[i] + 1) * 10
    # knn = KNeighborsClassifier(n_neighbors = 6)
    # knn.fit(X_train_pca, y_train)
    # y_pred = knn.predict(X_test_pca)
    # accuracy = accuracy_score(y_test, y_pred)


    pca = PCA(n_components=n_components, svd_solver="auto", whiten=True).fit(X_train)
    eigenfaces = pca.components_.reshape((n_components, h, w))

    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    acc, y_pred = accuracy_knn(X_train_pca, X_test_pca, y_train, y_test)
    print("For",(comp[i]+1)*10 , "pca components")
    print("Accuracy:", acc)

    accuracies.append(acc)

  return accuracies

accuracies = dif_accuracy(X_train, comp, X_test, y_train, y_test)

plt.plot((comp + 1) * 10, accuracies, marker='o')
plt.xlabel('Number of PCA Components')
plt.ylabel('Accuracy')
plt.title('Number of PCA Components vs Accuracy')
plt.grid(True)
plt.show()

n_faces = 10

plt.figure(figsize=(10, 5))
for i in range(n_faces):
  plt.subplot (2, 5, i + 1)
  plt.imshow(pca.components_[i].reshape(faces.images[0].shape), cmap='gray')
  plt.title(f"Eigenface {i+1}")
  plt.axis ('off')


plt.show()