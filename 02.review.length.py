import scipy
import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re
import math
import matplotlib.pyplot as plt

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

sentences_lengths = []
for sent in data_processed:
    sentences_lengths.append(len(sent.split()))

mean = np.mean(sentences_lengths)
var = np.var(sentences_lengths)
std = np.std(sentences_lengths)


# code without math
_, bins, _ = plt.hist(sentences_lengths, 100, density=1, color="green", alpha=0.7)
mu, sigma = scipy.stats.norm.fit(sentences_lengths)
best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)

# code with math
x = np.linspace(mean - 3 * math.sqrt(var), mean + 3 * math.sqrt(var), 500)
y = 1 / (std * math.sqrt(2 * np.pi)) * np.exp(-1 / 2 * ((x - mean) / std) ** 2)

plt.plot(x, y, color="blue")
plt.plot(bins, best_fit_line, color="orange")

# same!
plt.xlabel("Sentences length")
plt.ylabel("Density")

plt.title("Histogram Plot", fontweight="bold")
plt.show()
