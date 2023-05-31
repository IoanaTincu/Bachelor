import math
import xml.etree.ElementTree as ET
import os
import random
from nltk.corpus import stopwords
import spacy
from gensim.parsing.preprocessing import STOPWORDS
from spacy.lang.en import English
import heapq

from SampleFormat import SampleFormat


class TextDocumentsProcessing:

    def __init__(self, numberFiles):
        self.nlp = spacy.load('en_core_web_sm')
        self.spacyStopwords = spacy.lang.en.stop_words.STOP_WORDS
        self.stopwordsList = set(list(STOPWORDS) + list(self.spacyStopwords) + stopwords.words('english'))
        self.postags = ['NOUN', 'PROPN', 'VERB', 'ADJ']
        self.vocabulary = {}
        self.wordsInVocabulary = 0
        self.samples = []
        self.sumOfFrequencies = []
        self.topics = {}
        self.occurrencesOfTopics = 0
        self.occurrencesOfWords = {}
        self.numberFiles = numberFiles
        self.sumOfOccurrencesOfWords = {}
        self.mappedOccurrences = {}
        self.topicsOfWords = {}
        self.topicsOfSamples = []
        self.informationGains = {}
        self.mappedTopicsOfWords = {}
        self.actualNumberFiles = 0
        self.entropyDataset = 0
        self.selectedWords = []

    def parse_XML_file(self, xmlFile):
        parsedFile = ET.parse(xmlFile)
        root = parsedFile.getroot()

        title = root.find('title').text
        text = '\n'.join([p.text for p in root.find('text').findall('p')])
        topics = [code.attrib['code'] for code in root.findall(".//codes[@class='bip:topics:1.0']/code")]

        return title, text, topics

    def generate_files(self):
        path = r"C:\Users\Tincu\Downloads\Reuters\Reuters_7083"

        xmlFiles = [os.path.join(path, f) for f in os.listdir(path)]

        return random.sample(xmlFiles, self.numberFiles)

    def perform_feature_extraction(self, title, text, topics):
        textDocumentTokens = self.nlp(title + '\n' + text)

        sample = {}

        for token in textDocumentTokens:
            if not token.is_punct \
                    and not token.is_space \
                    and not token.lower_ in self.stopwordsList \
                    and token.is_alpha \
                    and token.pos_ in self.postags:
                processedToken = token.lemma_.lower()

                if processedToken not in self.vocabulary:
                    self.vocabulary[processedToken] = self.wordsInVocabulary
                    self.wordsInVocabulary += 1

                if self.vocabulary[processedToken] not in sample:
                    sample[self.vocabulary[processedToken]] = 1
                else:
                    sample[self.vocabulary[processedToken]] += 1

        for word in sample:
            if word not in self.occurrencesOfWords:
                self.occurrencesOfWords[word] = {sample[word]}
                self.sumOfOccurrencesOfWords[word] = sample[word]
            else:
                if sample[word] not in self.occurrencesOfWords[word]:
                    self.occurrencesOfWords[word].add(sample[word])
                    self.sumOfOccurrencesOfWords[word] += sample[word]

        self.samples.append(sample)

    def process_text_documents(self):
        samples = self.generate_files()
        # samples = ["C:\\Users\\Tincu\\Downloads\\Reuters\\Reuters_7083\\2822NEWS.XML"]
        # samples = ["C:\\Users\\Tincu\\Downloads\\Reuters\\Reuters_7083\\2504NEWS - Copy.XML",
        #            "C:\\Users\\Tincu\\Downloads\\Reuters\\Reuters_7083\\2538NEWS - Copy.XML",
        #            "C:\\Users\\Tincu\\Downloads\\Reuters\\Reuters_7083\\2775NEWS - Copy.XML"]

        for i in range(self.numberFiles):
            title, text, topics = self.parse_XML_file(samples[i])
            self.topicsOfSamples.append(topics)

            for topic in topics:
                self.occurrencesOfTopics += 1

                if topic not in self.topics:
                    self.topics[topic] = 1
                else:
                    self.topics[topic] += 1
                    if self.topics[topic] == self.numberFiles:
                        del self.topics[topic]
                        self.occurrencesOfTopics -= self.numberFiles

            self.perform_feature_extraction(title, text, topics)

        self.map_words()
        self.compute_entropy_of_data_set()
        self.compute_information_gains()
        self.perform_feature_selection()

        # dataset, numberFiles, attributes = self.convert_selected_words_to_sample_format()
        # print(dataset)

        # self.normalize_samples()
        return self.convert_selected_words_to_sample_format()

    def map_words(self):
        for word in self.vocabulary:
            lowerThanMean = 0
            greaterThanMean = 0
            numberZeros = 0
            mean = self.sumOfOccurrencesOfWords[self.vocabulary[word]] / len(
                self.occurrencesOfWords[self.vocabulary[word]])

            for j in range(len(self.samples)):
                if self.vocabulary[word] not in self.topicsOfWords:
                    self.topicsOfWords[self.vocabulary[word]] = {}
                    self.mappedTopicsOfWords[self.vocabulary[word]] = {}

                for topic in self.topicsOfSamples[j]:
                    if topic in self.topics:
                        if topic not in self.topicsOfWords[self.vocabulary[word]]:
                            self.topicsOfWords[self.vocabulary[word]][topic] = []
                            self.mappedTopicsOfWords[self.vocabulary[word]][topic] = []

                        if self.vocabulary[word] not in self.samples[j]:
                            self.topicsOfWords[self.vocabulary[word]][topic].append(0)
                            self.mappedTopicsOfWords[self.vocabulary[word]][topic].append(0)
                            numberZeros += 1
                        else:
                            self.topicsOfWords[self.vocabulary[word]][topic].append(self.samples[j][self.vocabulary[word]])

                            if self.samples[j][self.vocabulary[word]] <= mean:
                                lowerThanMean += 1
                                self.mappedTopicsOfWords[self.vocabulary[word]][topic].append(1)
                            else:
                                greaterThanMean += 1
                                self.mappedTopicsOfWords[self.vocabulary[word]][topic].append(2)

            self.mappedOccurrences[self.vocabulary[word]] = [numberZeros, lowerThanMean, greaterThanMean]

            if self.actualNumberFiles == 0:
                self.actualNumberFiles = numberZeros + lowerThanMean + greaterThanMean

    def compute_entropy_of_data_set(self):
        for topic in self.topics:
            probability = self.topics[topic] / self.occurrencesOfTopics
            self.entropyDataset += - probability * math.log2(probability)

    def compute_information_gains(self):
        for word in self.vocabulary:
            sum = 0
            probabilities = {}

            for i in range(3):
                entropy = 0

                if self.mappedOccurrences[self.vocabulary[word]][i] != 0:
                    for topic in self.topics:
                        probabilities[topic] = self.mappedTopicsOfWords[self.vocabulary[word]][topic].count(i)
                        probabilities[topic] /= self.mappedOccurrences[self.vocabulary[word]][i]

                        if probabilities[topic] != 0:
                            entropy += - probabilities[topic] * math.log2(probabilities[topic])

                    sum += self.mappedOccurrences[self.vocabulary[word]][i] / self.actualNumberFiles * entropy

            self.informationGains[self.vocabulary[word]] = self.entropyDataset - sum

    def perform_feature_selection(self):
        threshold = 0.9
        numberWordsToSelect = int(len(self.vocabulary) * threshold)
        self.selectedWords = heapq.nlargest(numberWordsToSelect, self.informationGains.items(), key=lambda x: x[1])

    def convert_selected_words_to_sample_format(self):
        words = [tuple[0] for tuple in self.selectedWords]
        dataset = [[] for _ in range(self.numberFiles)]

        for i in range(len(words)):
            for j in range(self.numberFiles):
                if words[i] in self.samples[j]:
                    dataset[j].append(SampleFormat(i, self.samples[j][words[i]]))

        return dataset, self.numberFiles, len(words)

    def normalize_samples(self):
        dataset, numberFiles, attributes = self.convert_selected_words_to_sample_format()

        for i in range(numberFiles):
            sumOfFrequencies = 0

            for word in dataset[i]:
                sumOfFrequencies += pow(word.frequencyOfAttribute, 2)

            for word in dataset[i]:
                word.frequencyOfAttribute = 3 * word.frequencyOfAttribute / math.sqrt(sumOfFrequencies)

        return dataset, numberFiles, attributes


# processing = TextDocumentsProcessing(3)
# processing.process_text_documents()
# print(processing.vocabulary.keys())
# print(processing.samples)
# print(processing.topics)
# print(processing.occurrencesOfTopics)
# print(processing.mappedOccurrences)
# print(processing.topicsOfWords)
# print(processing.mappedTopicsOfWords)
# print(processing.entropyDataset)
# print(processing.informationGains)
# print(processing.selectedWords)
