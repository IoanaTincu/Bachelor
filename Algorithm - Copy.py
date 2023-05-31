from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples

import numpy as np
from DataSet import TextDocumentsProcessing
from EuclideanDistance import calculate_distance
from main import read_arff_file
from SampleFormat import CandidSetFormat


class OpDbscan:

    def __init__(self, epsilon, n, minPts, numberFiles):
    # def __init__(self, epsilon, n, minPts):
    #     self.dataset, self.numberSamples, self.attributes = read_arff_file()

        self.processing = TextDocumentsProcessing(numberFiles)
        self.dataset, self.numberSamples, self.attributes = self.processing.process_text_documents()

        self.epsilon = epsilon
        self.n = n
        self.minPts = minPts
        self.clusters = [0] * self.numberSamples
        self.clusterNumber = 0
        self.visitedPointsInsideCluster = [0] * self.numberSamples
        self.currentCluster = [0] * self.numberSamples
        self.ptsInsideCurrentCluster = []
        self.accessiblePointsInTheOP = []
        self.candidSets = set()

    def find_epsilon_neighborhood(self, point):
        epsilonNeighborhood = []
        candidSet = []
        accessiblePointsInTheOP = []

        for i in range(self.numberSamples):
            if point == self.dataset[i]: continue

            distance = calculate_distance(point, self.dataset[i])

            if distance <= self.epsilon: epsilonNeighborhood.append(i)
            if distance <= self.n * self.epsilon:  # n*eps = Size of Operational Dataset
                if distance > (self.n - 1) * self.epsilon:
                    candidSet.append(i)
                else:
                    accessiblePointsInTheOP.append(i)

        return epsilonNeighborhood, candidSet, accessiblePointsInTheOP

    def calculate_distance_point_samplesOfOP(self, point):
        epsilonNeighborhood = []
        for i in range(len(self.accessiblePointsInTheOP)):
            if point == self.accessiblePointsInTheOP[i]: continue

            distance = calculate_distance(self.dataset[point], self.dataset[self.accessiblePointsInTheOP[i]])
            if distance <= self.epsilon: epsilonNeighborhood.append(self.accessiblePointsInTheOP[i])

        return epsilonNeighborhood

    def OP_DBSCAN_Algorithm(self):
        for i in range(self.numberSamples):
            if self.clusters[i] != 0: continue

            epsilonNeighborhood, candidSet, accessiblePointsInTheOP = self.find_epsilon_neighborhood(self.dataset[i])
            self.ptsInsideCurrentCluster = epsilonNeighborhood
            self.accessiblePointsInTheOP = accessiblePointsInTheOP

            if len(epsilonNeighborhood) < self.minPts: continue

            self.visitedPointsInsideCluster[i] = 1
            self.clusterNumber += 1
            self.currentCluster[i] = self.clusterNumber

            self.expand_cluster()
            self.move_to_cluster()

            self.visitedPointsInsideCluster = [0] * self.numberSamples
            self.currentCluster = [0] * self.numberSamples

            for point in candidSet:
                if self.clusters[point] == 0:
                    self.candidSets.add(CandidSetFormat(point, self.clusterNumber))

            self.update_OP()

        return self.clusters

    def expand_cluster(self):
        while len(self.ptsInsideCurrentCluster) > 0:
            point = self.ptsInsideCurrentCluster.pop()

            if self.visitedPointsInsideCluster[point] == 0:
                self.currentCluster[point] = self.clusterNumber
                self.visitedPointsInsideCluster[point] = 1

                if point in self.accessiblePointsInTheOP:
                    NB = self.calculate_distance_point_samplesOfOP(point)
                    if len(NB) >= self.minPts: self.ptsInsideCurrentCluster.extend(
                        NB)  # So, unlike Core Points, Non-Core Points can only join a cluster. They can not extend it further

    def move_to_cluster(self):
        counter = [0] * self.clusterNumber
        max = 0
        ptsInCurrentCluster = 0
        clusterNumberMax = 0
        currentClusterCopy = []

        for i in range(self.numberSamples):
            if self.currentCluster[i] != 0:
                ptsInCurrentCluster += 1
                currentClusterCopy.append(i)
            if self.currentCluster[i] != 0 and self.clusters[i] != 0:
                counter[self.clusters[i]] += 1
                if counter[self.clusters[i]] > max:
                    max = counter[self.clusters[i]]
                    clusterNumberMax = self.clusters[i]

        if not (max > ptsInCurrentCluster / 2): clusterNumberMax = self.clusterNumber
        for i in range(ptsInCurrentCluster):
            self.clusters[currentClusterCopy[i]] = clusterNumberMax

    def update_OP(self):
        nonCorePoints = set()

        while len(self.candidSets) > 0:
            candidSetPoint = self.candidSets.pop()
            point = candidSetPoint.point

            epsilonNeighborhood, candidSet, accessiblePointsInTheOP = self.find_epsilon_neighborhood(
                self.dataset[point])
            self.ptsInsideCurrentCluster = epsilonNeighborhood
            self.accessiblePointsInTheOP = accessiblePointsInTheOP

            if len(epsilonNeighborhood) < self.minPts:
                nonCorePoints.add(candidSetPoint)
                continue

            self.visitedPointsInsideCluster[point] = 1
            self.clusterNumber += 1
            self.currentCluster[point] = self.clusterNumber

            self.expand_cluster()
            self.move_to_cluster()

            self.visitedPointsInsideCluster = [0] * self.numberSamples
            self.currentCluster = [0] * self.numberSamples

            for point in candidSet:
                if self.clusters[point] == 0:
                    self.candidSets.add(CandidSetFormat(point, self.clusterNumber))

        while len(nonCorePoints) > 0:
            nonCore = nonCorePoints.pop()

            if self.clusters[nonCore.point] == 0:
                self.clusters[nonCore.point] = nonCore.motherCluster




def change_text_documents_into_numpy(samples, attributes):
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

def main():
    eps = 0.1
    while eps <= 4:
        algorithm = OpDbscan(eps, 2, 2, 2000)
        # algorithm = OpDbscan(1.9, 3, 2)
        clusters = algorithm.OP_DBSCAN_Algorithm()
        eps += 0.1

        indices = {}
        for i, point in enumerate(clusters):
            if point not in indices:
                indices[point] = [i]
            else:
                indices[point].append(i)

        print(len(indices))
        print(indices)

        badClassified = 0
        goodClassified = 0
        worstClassified = 0
        bestClassified = 0

        if len(set(clusters)) > 2:
            samples = change_text_documents_into_numpy(algorithm.dataset, algorithm.attributes)
            silhouetteIndex = silhouette_score(samples, clusters)
            silhouetteIndices = silhouette_samples(samples, clusters)
            DB = davies_bouldin_score(samples, clusters)

            for index in silhouetteIndices:
                if index >= -0.5 and index <= 0:
                    badClassified += 1

                if index > 0 and index <= 0.5:
                    goodClassified += 1

                if index >= -1 and index < -0.5:
                    worstClassified += 1

                if index > 0.5 and index <= 1:
                    bestClassified += 1

            print(str(eps) + ': ' + str(silhouetteIndex) + ' ' + str(badClassified) + ' ' + str(
                goodClassified) + ' ' + str(worstClassified) + ' ' + str(bestClassified) + ' ' + str(DB))
        else:
            print(str(eps) + ': ')


main()
