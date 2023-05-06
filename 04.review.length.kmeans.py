import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re

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
chunk_size = 30
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


data_processed = [preprocess_text(d) for d in data]
sentences_lengths = []
for sent in data_processed:
    sentences_lengths.append(len(sent.split()))
# Define the number of clusters
num_clusters = 3

# Define the number of iterations
num_iterations = 100

centroids = [
    sentences_lengths[0],
    sentences_lengths[(total_documents - 1) // 3],
    sentences_lengths[total_documents - 10],
]
print(f"Initial centroid: {centroids}")
cluster_sentences = []
points = []

for i in range(num_iterations):
    distance_matrix = np.zeros((total_documents, num_clusters))
    points = []
    for m in range(len(sentences_lengths)):
        for n in range(len(centroids)):
            distance_matrix[m, n] = abs(sentences_lengths[m] - centroids[n])

        points.append((m, np.argmin(distance_matrix[m, :])))
    centroids = []

    # Recalculate the centroids
    for k in range(num_clusters):
        # Get all values of sentence_length that belong to the ith cluster
        keys_with_value_i = [x[0] for x in points if x[1] == k]
        #         print("Here:",keys_with_value_i)
        sum_length = 0
        for index in keys_with_value_i:
            sum_length += sentences_lengths[index]
        centroid = sum_length / len(keys_with_value_i)
        centroids.append(centroid)

    # At the final iteration, return a list of sentence length and the cluster it belongs to
    if i == num_iterations - 1:
        print("Here")
        for j in range(len(points)):
            cluster_sentences.append(
                (
                    data_processed[points[j][0]],
                    len(data_processed[points[j][0]].split()),
                    points[j][1],
                )
            )

print(f"Final centroid: {centroids}")
# print(f"Sentences-length-cluster_belogs: {cluster_sentences}")
