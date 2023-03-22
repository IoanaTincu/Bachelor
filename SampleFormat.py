class SampleFormat:

    def __init__(self, index, occurrences):
        self.indexOfAttribute = index
        self.frequencyOfAttribute = occurrences

    def __str__(self):
        return '({0}, {1})'.format(self.indexOfAttribute, self.frequencyOfAttribute)

    def __repr__(self):
        return str(self)


class CandidSetFormat:

    def __init__(self, point, clusterNumber):
        self.point = point
        self.motherCluster = clusterNumber

    def __str__(self):
        return '({0}, {1})'.format(self.point, self.motherCluster)

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.point)

    def __eq__(self, other):
        return self.point == other.point
