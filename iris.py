'''Sample script using K Nearest Neighbors on the Iris datset'''

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

# scikit-learn libraries
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# load the data set TR
iris_set = pd.read_csv('datasets//iris.csv')
iris_dataset = pd.DataFrame(iris_set)
print(iris_dataset.head(5))

# clean and split the data set TR
iris_dataset = shuffle(iris_dataset)

data_encoder = preprocessing.LabelEncoder()
iris_dataset['variety'] = data_encoder.fit_transform(iris_dataset['variety'])


# compare features and determine relations between features and final classification TR
def make_plots(some_data):
    some_data.plot(kind='scatter', x='sepal.length', y='variety', color='purple')
    plt.savefig('plots\\sepal_length.png')

    some_data.plot(kind='scatter', x='sepal.width', y='variety', color='purple')
    plt.savefig('plots\\sepal_width.png')

    some_data.plot(kind='scatter', x='petal.length', y='variety', color='purple')
    plt.savefig('plots\\petal_length.png')

    some_data.plot(kind='scatter', x='petal.width', y='variety', color='purple')
    plt.savefig('plots\\petal_width.png')
    plt.show()


# now use K nearest neighbors to train and test
def create_model(some_data):
    X = some_data.iloc[:, :-1]
    y = some_data['variety']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

    # determine the ideal number of k
    k_range = range(1, 18)
    scores = {}
    scores_list = []

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        scores[k] = metrics.accuracy_score(y_test, y_pred)
        scores_list.append(metrics.accuracy_score(y_test, y_pred))

    plt.plot(k_range, scores_list)
    plt.xlabel('Value of K')
    plt.ylabel('Accuracy')
    plt.savefig('plots\\ideal_k.png')
    # based on these results we received perfect accuracy at 11 clusters, and 96% at 3,7, and 9 clusters

    # my final k
    classes = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    new_x = [[3, 4, 5, 2],
             [5, 4, 2, 2]]
    knn_final = KNeighborsClassifier(n_neighbors=3)
    knn_final.fit(X_train, y_train)
    y_final_pred = knn_final.predict(new_x)

    print(classes[y_final_pred[0]])
    print(classes[y_final_pred[1]])


# make_plots(iris_dataset)
create_model(iris_dataset)
