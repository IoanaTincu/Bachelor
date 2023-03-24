import xml.etree.ElementTree as ET
import os
import random
from nltk.corpus import stopwords
import spacy
from gensim.parsing.preprocessing import STOPWORDS
from spacy.lang.en import English


class TextDocumentsProcessing:

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.spacyStopwords = spacy.lang.en.stop_words.STOP_WORDS
        self.stopwordsList = set(list(STOPWORDS) + list(self.spacyStopwords) + stopwords.words('english'))
        self.postags = ['NOUN', 'PROPN', 'VERB', 'ADJ']
        self.vocabulary = set()
        self.samples = []


    def parse_XML_file(self, xmlFile):
        parsedFile = ET.parse(xmlFile)
        root = parsedFile.getroot()

        title = root.find('title').text
        text = '\n'.join([p.text for p in root.find('text').findall('p')])
        topics = [code.attrib['code'] for code in root.findall(".//codes[@class='bip:topics:1.0']/code")]

        return title, text, topics


    def generate_files(self, numberFiles):
        path = r"C:\Users\Tincu\Downloads\Reuters\Reuters_34\Training"

        xmlFiles = [os.path.join(path, f) for f in os.listdir(path)]

        return random.sample(xmlFiles, numberFiles)


    def perform_feature_extraction(self, title, text):
        textDocumentTokens = self.nlp(title + '\n' + text)

        sample = {}

        for token in textDocumentTokens:
            if not token.is_punct \
                    and not token.is_space \
                    and not token.lower_ in self.stopwordsList \
                    and token.is_alpha \
                    and token.pos_ in self.postags:
                processedToken = token.lemma_.lower()

                self.vocabulary.add(processedToken)

                if processedToken not in sample:
                    sample[processedToken] = 1
                else:
                    sample[processedToken] += 1

        self.samples.append(sample)


    def process_text_documents(self, numberFiles):
        samples = self.generate_files(numberFiles)

        for i in range(numberFiles):
            title, text, topics = self.parse_XML_file(samples[i])

            self.perform_feature_extraction(title, text)




processing = TextDocumentsProcessing()
processing.process_text_documents(1)
print(processing.vocabulary)
print(processing.samples)
