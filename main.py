# ================================
# Automatic Duplicate Question Detection
# Fixed & Stable Version
# ================================

import pandas as pd
import numpy as np
import gensim
import gensim.corpora as corpora
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings("ignore")

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# -------------------------------
# PARAMETERS (SAFE FOR LAPTOP)
# -------------------------------
N1 = 100   # number of past questions
N2 = 30    # number of duplicate questions
NUM_TOPICS = 50

# -------------------------------
# PREPROCESSING FUNCTIONS
# -------------------------------
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = BeautifulSoup(str(text), "html.parser").get_text()
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [ps.stem(w) for w in tokens if w.isalpha() and w not in stop_words]
    return " ".join(tokens)

# -------------------------------
# LOAD DATASET
# -------------------------------
df = pd.read_csv("QueryResults.csv")

past_df = df[['PastQuesTitle', 'PastQuesBody', 'PastQuesTags']].iloc[:N1]
dup_df  = df[['DuplicateQuesTitle', 'DuplicateQuesBody', 'DuplicateQuesTags']].iloc[:N2]

# Apply preprocessing
for col in past_df.columns:
    past_df[col] = past_df[col].apply(clean_text)

for col in dup_df.columns:
    dup_df[col] = dup_df[col].apply(clean_text)

print("✔ Preprocessing completed")

# -------------------------------
# TF-IDF VECTOR REPRESENTATION
# -------------------------------
past_text = past_df['PastQuesTitle'] + " " + past_df['PastQuesBody']
dup_text  = dup_df['DuplicateQuesTitle'] + " " + dup_df['DuplicateQuesBody']

tfidf = TfidfVectorizer(max_features=5000)
tfidf_past = tfidf.fit_transform(past_text)
tfidf_dup  = tfidf.transform(dup_text)

tfidf_sim = cosine_similarity(tfidf_past, tfidf_dup)

print("✔ TF-IDF similarity computed")

# -------------------------------
# LDA TOPIC MODELING
# -------------------------------
texts = [doc.split() for doc in past_text]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

lda_model = gensim.models.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=NUM_TOPICS,
    passes=5
)

def get_topic_vector(text):
    bow = dictionary.doc2bow(text.split())
    topics = lda_model.get_document_topics(bow, minimum_probability=0)
    return np.array([t[1] for t in topics])

lda_past = np.array([get_topic_vector(t) for t in past_text])
lda_dup  = np.array([get_topic_vector(t) for t in dup_text])

lda_sim = cosine_similarity(lda_past, lda_dup)

print("✔ LDA similarity computed")

# -------------------------------
# FINAL SIMILARITY (WEIGHTED)
# -------------------------------
ALPHA = 0.7   # TF-IDF weight
BETA  = 0.3   # LDA weight

final_similarity = ALPHA * tfidf_sim + BETA * lda_sim

# -------------------------------
# EVALUATION : Recall@K
# -------------------------------
def recall_at_k(sim_matrix, k):
    correct = 0
    for i in range(sim_matrix.shape[1]):
        top_k = np.argsort(sim_matrix[:, i])[::-1][:k]
        if i in top_k:
            correct += 1
    return correct / sim_matrix.shape[1]

for k in [1, 5, 10]:
    print(f"Recall@{k}: {recall_at_k(final_similarity, k):.2f}")

print("✔ Execution completed successfully")
