# Link notebook: https://www.kaggle.com/jinkaido/labwork-data-mining
import pandas as pd
import numpy as np
import json
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
import re
import math
from collections import Counter

data = []
word_set = []
commonly_review_words = [
    "delicious",
    "amazing",
    "excellent",
    "great",
    "good",
    "tasty",
    "yummy",
    "fantastic",
    "perfect",
    "friendly",
    "clean",
    "comfortable",
    "cozy",
    "spacious",
    "beautiful",
    "lovely",
    "awesome",
    "wonderful",
    "nice",
    "incredible",
    "impressive",
    "outstanding",
    "satisfying",
    "fresh",
    "quality",
    "value",
    "affordable",
    "reasonable",
    "fast",
    "quick",
    "efficient",
    "attentive",
    "helpful",
    "polite",
    "professional",
    "knowledgeable",
    "accommodating",
    "satisfactory",
    "enjoyable",
    "pleasant",
    "memorable",
    "unique",
    "authentic",
    "ambiance",
    "atmosphere",
    "vibe",
    "experience",
    "like",
    "get",
    "recommend",
    "must-try",
    "must-visit",
    "top-notch",
    "highly-rated",
    "popular",
    "famous",
    "iconic",
    "legendary",
]
read_chunk = 0

chunk_total = 5
chunk_size = 10
total_documents = chunk_size * chunk_total

chunks = pd.read_json(
    "/kaggle/input/yelp-dataset/yelp_academic_dataset_review.json",
    lines=True,
    chunksize=chunk_size,
)
for c in chunks:
    data.extend(c["text"].astype(str).tolist())
    read_chunk += 1
    if read_chunk == chunk_total:
        break


def preprocess_text(sentence):
    sentence = re.sub(r"\W+", " ", sentence)
    sentence = re.sub("[^a-zA-Z]+", " ", sentence)
    sentence = sentence.lower()
    sentence = sentence.strip()
    tokenizer = RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(sentence)
    words = []
    for w in tokens:
        if w not in stopwords.words("english") or w not in commonly_review_words:
            words.append(w)

    for word in words:
        word_set.append(word)
    return " ".join(words)


data_processed = [preprocess_text(d) for d in data]  # Preprocessed sentence
word_set = set(word_set)  # Set that contains all the vocabulary
index_dict = {}  # Dictionary to store index for each word
idf_dict = {}  # Dictionary to store idf value
tf_idf_matrix = []  # TF-IDF matrix

for i, word in enumerate(word_set):
    index_dict[word] = i


for word in word_set:
    idf_dict[word] = 0
    for sent in data_processed:
        if word in set(sent.split()):
            idf_dict[word] += 1
for key in idf_dict.keys():
    idf_dict[key] = math.log(total_documents / idf_dict[key])


def termfreq(sentence, word):
    return Counter(sentence.split())[word] / len(sentence.split())


def tf_idf(sentence):
    tf_idf_vec = np.zeros((len(word_set),))
    for word in sentence.split():
        tf = termfreq(sentence, word)
        idf = idf_dict[word]
        value = tf * idf
        tf_idf_vec[index_dict[word]] = value
    return tf_idf_vec


for sent in data_processed:
    vec = tf_idf(sent)
    tf_idf_matrix.append(vec)
