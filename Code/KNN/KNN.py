import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets                 # install by   pip install scikit-learn    #https://scikit-learn.org/stable/
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# The iris dataset is a classic and very easy multi-class classification dataset.
# - Classes: 3
# - Samples per class: 50
# - Samples total: 150
# - Dimensionality: 4
# - Features: real, positive
iris = datasets.load_iris()

# The dataset
iris_X = iris.data

# The labels of dataset above
iris_y = iris.target

print ('Number of classes: %d'%len(np.unique(iris_y)))
print ('Number of data points: %d' %len(iris_y))

# Extract all data with the label is class 0
#X0 = iris_X[iris_y == 0,:]
#print ('\nSamples from class 0:\n', X0[:5,:])

# Extract all data with the label is class 1
#X1 = iris_X[iris_y == 1,:]
#print ('\nSamples from class 1:\n', X1[:5,:])

# Extract all data with the label is class 2
#X2 = iris_X[iris_y == 2,:]
#print ('\nSamples from class 2:\n', X2[:5,:])


# Just get a portion of data set for training and testing.
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=50)

#print ("Training size: %d" %len(y_train))
#print ("Test size    : %d" %len(y_test))


#Declare the algorithm with K=2 neighbors (default=5) and use Minkowski metric with p=2 for special case euclidean_distance, p=1 for manhattan_distance 
clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2)

# Implement the KNN with training dataset to get model
clf.fit(X_train, y_train)

# Apply the model with test data.
y_pred = clf.predict(X_test)

print ("Print results for 20 test data points:")
print ("Predicted labels: ", y_pred[20:40])
print ("Ground truth    : ", y_test[20:40])
print ("Accuracy of 1NN: %.2f %%" %(100*accuracy_score(y_test, y_pred)))


clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2, weights = 'distance')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print ("Accuracy of 10NN (1/distance weights): %.2f %%" %(100*accuracy_score(y_test, y_pred)))


def myweight(distances):
    """ Customize the distance definition
    distances: raw distance 
    Return: distance
    """
    sigma2 = .5 # we can change this number
    return np.exp(-distances**2/sigma2)

clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2, weights = myweight)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print ("Accuracy of 10NN (customized weights): %.2f %%" %(100*accuracy_score(y_test, y_pred)))
