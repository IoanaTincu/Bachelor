import pandas as pd
import math

from SampleFormat import SampleFormat


def read_arff_file():
    df = pd.read_csv(r"C:\Users\Tincu\Downloads\Reuters - clasification\MultiClass_Training_SVM_1309.0.arff")

    numberSamples = df.columns[0].split()[1]
    attributes = df[df.columns[0]][0].split()[1]
    topics = df[df.columns[0]][1].split()[1]

    linesInFile = df.shape[0]
    samples = []

    for i in range(linesInFile - 1 - int(numberSamples) + 1,
                   linesInFile):  # linesInFile-1-int(samples)+1 : line of first sample
        row = df[df.columns[0]][i].split('#')[0].split()
        topicsSample = df[df.columns[0]][i].split('#')[1]
        sample = []
        #sumOfFrequencies = 0

        #for data in row:
            #sumOfFrequencies += int(data.split(':')[1])

        for data in row:
            #sample.append(SampleFormat(data.split(':')[0], int(data.split(':')[1]) / sumOfFrequencies))

            sample.append(SampleFormat(data.split(':')[0], int(data.split(':')[1])))

            #sample.append(SampleFormat(data.split(':')[0], 1))

            #occurrences = int(data.split(':')[1])
            #sample.append(SampleFormat(data.split(':')[0], 1 + math.log10(1 + math.log10(occurrences))))

        samples.append(sample)

    return samples, int(numberSamples), int(attributes)


