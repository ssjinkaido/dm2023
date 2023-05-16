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
chunk_size = 100
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

bandwidth = 20
threshold = 0.005
centroids = []
sentences_lengths = []


# https://en.wikipedia.org/wiki/Mean_shift
def flat_kernel(distance, bandwidth):
    if distance < bandwidth:
        return 1
    else:
        return 0


# cannot find the parameters to make gaussian kernel works
for sent in data_processed:
    sentences_lengths.append(len(sent.split()))

for i in range(len(sentences_lengths)):
    current_centroid = sentences_lengths[i]
    while True:
        total_distance = 0
        scale_factor = 0
        for j in range(len(sentences_lengths)):
            distance = current_centroid - sentences_lengths[j]
            #             if distance<=bandwidth:
            weight = flat_kernel(distance, bandwidth)
            if weight == 0:
                continue
            total_distance += weight * sentences_lengths[j]
            scale_factor += weight

        new_centroid = total_distance / scale_factor
        if (new_centroid - current_centroid) < threshold:
            centroids.append(new_centroid)
            break
        else:
            current_centroid = new_centroid
centroids = set(centroids)

print(f"Final centroids: {centroids}")
