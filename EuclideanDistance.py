import math

from DataSet import TextDocumentsProcessing
from main import read_arff_file


def calculate_distance(textDocument1, textDocument2):
    sumOfSquares = 0
    i = j = 0

    while i < len(textDocument1) and j < len(textDocument2):
        if int(textDocument1[i].indexOfAttribute) < int(textDocument2[j].indexOfAttribute):
            sumOfSquares += pow(int(textDocument1[i].frequencyOfAttribute), 2)
            i += 1

        elif int(textDocument1[i].indexOfAttribute) > int(textDocument2[j].indexOfAttribute):
            sumOfSquares += pow(int(textDocument2[j].frequencyOfAttribute), 2)
            j += 1

        else:
            sumOfSquares += pow(int(textDocument1[i].frequencyOfAttribute) - int(textDocument2[j].frequencyOfAttribute),
                                2)
            i += 1
            j += 1

    while i < len(textDocument1):
        sumOfSquares += pow(int(textDocument1[i].frequencyOfAttribute), 2)
        i += 1

    while j < len(textDocument2):
        sumOfSquares += pow(int(textDocument2[j].frequencyOfAttribute), 2)
        j += 1

    return math.sqrt(sumOfSquares)




# processing = TextDocumentsProcessing(27)
# dataset, numberSamples, attributes = processing.process_text_documents()
# print(len(processing.topics))
# print(processing.topics)
#
# dataset, numberSamples, attributes = read_arff_file()
#
#
# def minimum_distance():
#     minimum = 100
#     maximum = -1
#
#     for i in range(len(dataset)):
#         for j in range(i + 1, len(dataset)):
#             distance = calculate_distance(dataset[i], dataset[j])
#
#             if distance < minimum: minimum = distance
#             if distance > maximum: maximum = distance
#
#             # print(distance)
#
#     print('minimum ' + str(minimum) + 'maximum ' + str(maximum))
#
#
# minimum_distance()
