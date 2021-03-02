# File: KNN.py

import time
import numpy as np
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

import warnings

warnings.filterwarnings("ignore")


def cm_accuracy(cm):
    diagonal_sum = cm.trace()
    sum_of_all_elements = cm.sum()
    return diagonal_sum / sum_of_all_elements


def _accuracy(y_tst, y_predicted):
    """ Calculate the Accuracy """
    acc = np.sum(y_tst == y_predicted) / len(y_tst)

    return acc


def draw_plot(xs, ys):
    cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    plt.figure()
    plt.scatter(xs[:, 0], xs[:, 1],
                c=ys,
                cmap=cmap,
                edgecolor='k',
                s=25)
    plt.show()


"""
    There are various methods for calculating the distance
    between the points, of which the most commonly known
    methods are –
        1. Euclidean,
        2. Manhattan, 
        3. Minkowski, and
        3. Hamming distance (for categorical).
"""


def manhattan_distance(point1, point2):
    """ Manhattan Distance """
    """ distance = sum|xi - yi| """
    return np.sum(np.abs(point1 - point2))


def euclidean_distances(point1, point2):
    """ Euclidean Distance """
    """
    We usually use Euclidean distance to calculate the nearest
    neighbor. If we have two points (x1, y1) and (x2, y2). 
    The formula for Euclidean distance (d) will be: 
        distance = sqrt((x1 - x2)² + (y1 - y2)²)
   """
    return np.sqrt(np.sum(np.power((point1 - point2), 2)))


class KNN:
    """ K Nearest Neighbor Algorithm """
    """ Just to understand the math under the hood """

    def __init__(self, n_neighbors=3, p=2):
        """ KNN Constructor """
        """
        K represents the number of the nearest neighbors
        that we used to classify new data points.
        """
        self.k = n_neighbors
        self.p = p

    def fit(self, x_trn, y_trn):
        """ Fitting Data """
        self.X_train = x_trn
        self.y_train = y_trn

    def predict(self, x_tst):
        """ Predict the classes of the test data """
        predicted_labels = [self.predict_labels(x)
                            for x in x_tst]
        return np.array(predicted_labels)

    def predict_labels(self, point1):
        """ Predict the common class for a point """
        # Calculate the Euclidean distance
        distances_list = [self.minkowski_distance(point1, point2)
                          for point2 in self.X_train]

        """
        Once the distance of a new observation from the points in
        our training set has been measured, the next step is to
        pick the closest points.
        The number of points to be considered is defined by the
        value of k.
        """
        # Return K index of the smallest distances
        k_index = np.argsort(distances_list)[:self.k]

        # VOTING for the outcome:
        # Find the associated labels
        k_index_labels = [self.y_train[i] for i in k_index]
        # Get the most common class - the final outcome
        most_common_label = Counter(k_index_labels).most_common(1)

        return most_common_label[0][0]

    def n_root(self, value):
        root_value = 1 / np.float64(self.p)
        return np.round(np.float64(value) ** np.float64(root_value), 3)

    def minkowski_distance(self, point1, point2):
        return self.n_root(np.sum(np.power(np.abs(point1 - point2), self.p)))


def sklearn_knn(X_train, X_test, y_train, y_test):
    # Create KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5,
                               # metric='minkowski', # default
                               p=2)
    # When p = 1 --> manhattan_distance
    # When p = 2 --> euclidean_distance
    # Fit model using training data
    knn.fit(X_train, y_train)
    # Predict on the training sets
    y_predictions = knn.predict(X_test)

    # Test accuracy using accuracy_score()
    acc = accuracy_score(y_test, y_predictions)
    print("\tAccuracy Score: {%.2f%%}" % (acc * 100.0))

    # Test accuracy using confusion_matrix
    cm = confusion_matrix(y_test, y_predictions)
    cm_acc = cm_accuracy(cm)
    print("\tConfusion Matrix Accuracy Score: {%.2f%%}" % (cm_acc * 100.0))


def programmed_knn(X_train, X_test, y_train, y_test):
    K = 5
    p = 2
    # When p = 1 --> manhattan_distance
    # When p = 2 --> euclidean_distance
    """
    Choosing the right value of K is called parameter tuning and 
    it’s necessary for better results.
        - K = sqrt (total number of data points).
        - Odd value of K is always selected to avoid confusion between 2 
        classes.
    """
    # Create KNN classifier
    clf = KNN(n_neighbors=K, p=p)
    # Fit x and y training data
    clf.fit(X_train, y_train)
    # Predict the labels of the y testing data
    y_predictions = clf.predict(X_test)
    # What is the accuracy rate of the model
    print("\tAccuracy Score: {%.2f%%}" % (_accuracy(y_test, y_predictions) * 100.0))


if __name__ == '__main__':
    # Load Iris data from sklearn datasets
    iris = datasets.load_iris()
    # Create X and y data
    X, y = iris.data, iris.target
    # draw_plot(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=1234)
    """
        Knn is simple to implement and mostly used for classification.
        Knn is easy to interpret.
        Knn for large datasets requires a lot of memory and gets slow
            to predict because of distance calculations
        Knn accuracy can be broken down if there are many predictors
        Knn doesn't generate insights
    """
    try:
        while True:
            print(f"\n Choose a strategy?"
                  f"\n 1 : KNN manually programmed!"
                  f"\n 2 : KNN using sklearn library!"
                  f"\n  ..... Press any other key to EXIT.")

            choice = int(input("\n Enter your choice: "))

            start = time.time()  # --------------------------------------------

            if choice == 1:
                programmed_knn(X_train, X_test, y_train, y_test)
            elif choice == 2:
                # Test Knn using sklearn library
                sklearn_knn(X_train, X_test, y_train, y_test)
            else:
                print('..... Exit .....')
                break

            end = time.time()  # ----------------------------------------------
            print('Execution Time: {%f}' % ((end - start) / 1000) + ' seconds.')
    except:
        pass
