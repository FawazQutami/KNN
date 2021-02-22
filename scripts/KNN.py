# File: KNN.py

"""
   KNN: K Nearest Neighbor.
   KNN is one of the simplest forms of machine learning
   algorithms mostly used for classification.
   We can use KNN if the dataset is small, well-labeled,
   and noise-free.
"""

import numpy as np
from collections import Counter

from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


"""
    There are various methods for calculating the distance
    between the points, of which the most commonly known
    methods are –
        1. Euclidian,
        2. Manhattan (for continuous) and
        3. Hamming distance (for categorical).
    """
def euclidean_distances(point1, point2):
    """
    We usually use Euclidean distance to calculate the nearest
    neighbor. If we have two points (x1, y1) and (x2, y2). 
    The formula for Euclidean distance (d) will be: 
        distance = sqrt((x1-x2)²+(y1-y2)²)
    """
    return np.sqrt(np.sum((point1 - point2) ** 2))


class KNN:

    def __init__(self, k=3):
        """
        Constructor
        K represents the number of the nearest neighbors
        that we used to classify new data points.
        """
        self.k = k

    def fit(self, _X, _y):
        self.X_train = _X
        self.y_train = _y

    def predict(self, x_test):
        predicted_labels = [self.predict_labels(x) for x in x_test]
        return np.array(predicted_labels)

    def predict_labels(self, point1):
        # We try to get the smallest Euclidean distance and based
        # on the number of smaller distances we perform our
        # calculation.
        distances_List = [euclidean_distances(point1, point2) for point2 in self.X_train]
        
        """
        Once the distance of a new observation from the points in
        our training set has been measured, the next step is to
        pick the closest points.
        The number of points to be considered is defined by the
        value of k.
        """
        # Return K indexs of the smallest distances
        k_index = np.argsort(distances_List)[:self.k]
        
        ## VOTING for the outcome:
        # Find the associated labels
        k_index_labels = [self.y_train[i] for i in k_index]    
        # Get the most common class - the final outcome
        most_common_label = Counter(k_index_labels).most_common(1)
        return most_common_label[0][0]


def accuracy(y_test, y_prediction):
    acc = np.sum(y_test == y_prediction) / len(y_test)
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


if __name__ == '__main__':
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=1234)

    K = 5
    """
    Choosing the right value of K is called parameter tuning and 
    it’s necessary for better results.
        - K = sqrt (total number of data points).
        - Odd value of K is always selected to avoid confusion between 2 
        classes.
    """
    clf = KNN(k=K)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print("KNN Accuracy is: ", accuracy(y_test, predictions))

    draw_plot(X, y)
