import xml.etree.ElementTree as ET
import os
import random
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from gensim.parsing.preprocessing import STOPWORDS


def parse_XML_file(xmlFile):
    parsedFile = ET.parse(xmlFile)
    root = parsedFile.getroot()

    title = root.find('title').text
    text = '\n'.join([p.text for p in root.find('text').findall('p')])
    topics = [code.attrib['code'] for code in root.findall(".//codes[@class='bip:topics:1.0']/code")]

    return title, text, topics


def generate_files(numberFiles):
    path = r"C:\Users\Tincu\Downloads\Reuters\Reuters_34\Training"

    xmlFiles = [os.path.join(path, f) for f in os.listdir(path)]

    return random.sample(xmlFiles, numberFiles)


def perform_feature_extraction(title, text):
    tokens = word_tokenize(title + text)

    nlp = spacy.load('en_core_web_sm')
    spacyStopwords = spacy.lang.en.stop_words.STOP_WORDS

    stopwordsList = set(list(STOPWORDS) + list(spacyStopwords) + list(nlp.Defaults.stop_words) + stopwords.words('english'))
    if set(spacyStopwords) == set(nlp.Defaults.stop_words): print("equal")
    else: print("not equal")

    lemmatizer = WordNetLemmatizer()

    lemmatizedTokens = []
    for token in tokens:
        token = token.lower()
        if token not in stopwordsList:
            token = lemmatizer.lemmatize(token)
            lemmatizedTokens.append(token)

    return tokens




xmlFile = generate_files(1)
print(xmlFile)

title, text, topics = parse_XML_file(xmlFile[0])
print(title)
print(text)
print(topics)
print(perform_feature_extraction(title, text))
