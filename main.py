import argparse
from typing import Tuple

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from tsk.classifier import Classifier

parser = argparse.ArgumentParser(description='Takagi-Sugeno fuzzy system for FEHI')

parser.add_argument('--dataset', type=str, help='Dataset to use in the experiment')
parser.add_argument('--n_cluster', type=int, help='Number of clusters for C-Means clustering')


def parse_dataset(path: str) -> Tuple:
    """
    Load CSV file from storage and parse the data inside.

    :param path: (str) Path to the CSV file
    :return: (Tuple) X and y values extracted from the CSV file
    """
    with open(path, 'r') as f:
        data = f.readlines()

    clean_rows = [row.strip().split(',') for row in data]
    clean_rows = np.array([list(map(float, row)) for row in clean_rows])

    return clean_rows[:, :-1], clean_rows[:, -1]


def main():
    """Entry point of the application"""
    flags = parser.parse_args()

    # load and shuffle data
    x, y = parse_dataset(flags.dataset)
    x, y = shuffle(x, y)

    print(f'Loaded dataset from: {flags.dataset}')

    # prepare train/test split
    x_train = x[:400]
    y_train = y[:400]

    x_test = x[400:]
    y_test = y[400:]

    print(f'Number of training samples: {len(x_train)}')
    print(f'Number of test samples: {len(x_test)}')

    # fit the fuzzy classifier
    cls = Classifier()

    print('Fitting classifier to data:')

    cls.fit(x_train, y_train)

    # predict
    print('Predicting unseen data:')
    y_pred = cls.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'\taccuracy: {accuracy}')


if __name__ == '__main__':
    main()
