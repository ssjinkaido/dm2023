import pandas as pd
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


data_processed = [preprocess_text(d) for d in data]

clusters = []
max_length = 0
for sent in data_processed:
    clusters.append([(sent, len(sent.split()))])


def calc_dist_between_cluster(list_clusters_i, list_clusters_j):
    min_dist = 9999
    for i in list_clusters_i:
        for j in list_clusters_j:
            min_dist = min(abs(i[1] - j[1]), min_dist)
    return min_dist


while len(clusters) > 3:
    min_dist = 9999
    index_i = 0
    index_j = 0
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            min_dist_between_cluster = calc_dist_between_cluster(
                clusters[i], clusters[j]
            )
            if min_dist_between_cluster < min_dist:
                index_i = i
                index_j = j
                min_dist = min_dist_between_cluster
    for i in range(len(clusters[index_j])):
        clusters[index_i].append(clusters[index_j][i])
    del clusters[index_j]

for i in range(len(clusters)):
    print(f"Length cluster {i}: {len(clusters[i])}")
