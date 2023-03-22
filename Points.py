import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
import seaborn as sns
from main import read_arff_file



def get_2DPoints():
    return np.array([(1, 1), (3 / 2, 4), (2, 2), (3, 1 / 5), (5 / 2, 1.1), (3, 1.5),
                     (3.2, 1.3), (3.3, 1.5), (3.2, 1.5), (3.8, 2), (4, 1.8),
                     (4, 1.6), (3.5, 2), (4.1, 2), (4.3, 1.6), (4.5, 2), (4.7, 2.2),
                     (4.7, 1.3), (4.6, 2.8), (3.9, 2.7), (5, 1.4), (5.2, 2),
                     (5.4, 2.8), (5.6, 3), (5.8, 2.8), (5.9, 1.5), (6, 3.4),
                     (6.2, 3.6), (4.8, 3), (6, 3), (6, 2.8), (6.1, 2.6), (6.3, 2.3),
                     (6, 2.2), (6.2, 2), (6.4, 1.8), (6.5, 2), (6.6, 1.8), (6.7, 1.6),
                     (6.8, 1.7), (6.9, 1.5), (7, 1.4), (7.1, 1.3), (7.4, 1.5),
                     (7.5, 1), (7.7, 1.2), (7.6, 1.1), (7.6, 1.8), (7.6, 2),
                     (7.7, 1.4), (7.8, 1.6), (8, 1.8), (8.1, 1.7), (8.2, 1.8),
                     (8.2, 1.9), (8, 3), (8.2, 2.9), (8.4, 3.2), (8.6, 3.7),
                     (8.2, 3), (8.5, 3.8), (8.1, 4), (8.4, 3.7), (9, 3.6),
                     (9.1, 3.9), (9.3, 4), (9.5, 4.2), (9.7, 4), (9.6, 4.4), (9.8, 4.7),
                     (9.9, 5), (10, 4.8), (10, 6), (10.2, 6.2)])


def plot_2DPoints():
    points = get_2DPoints()

    algorithm = DBSCAN(eps=0.9, min_samples=2)
    clusters = algorithm.fit_predict(points)

    numberClusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    colours = sns.color_palette('bright', numberClusters)

    coloursClusters = [colours[x] if x >= 0 else (0.5, 0.5, 0.5) for x in clusters]
    plt.scatter(points[:, 0], points[:, 1], c=coloursClusters)

    plt.show()

#plot_2DPoints()


def change_text_documents_into_numpy():
    samples, numberSamples, attributes = read_arff_file()
    textDocuments = []

    for sample in samples:
        data = []
        i, j = 0, 0

        while i < attributes and j < len(sample):
            if i < int(sample[j].indexOfAttribute):
                data.append(0)
                i += 1
            else:
                data.append(sample[j].frequencyOfAttribute)
                i += 1
                j += 1

        while i < attributes:
            data.append(0)
            i += 1

        textDocuments.append(data)

    return np.array(textDocuments)


def cluster_text_documents():
    textDocuments = change_text_documents_into_numpy()
    algorithm = DBSCAN(eps=0.2, min_samples=4)
    clusters = algorithm.fit_predict(textDocuments)

    print(set(clusters))
    return clusters

cluster_text_documents()

