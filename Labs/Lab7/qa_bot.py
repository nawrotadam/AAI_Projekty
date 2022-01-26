# dataset source
# https://www.kaggle.com/rtatman/questionanswer-dataset

from flair.embeddings import FlairEmbeddings, StackedEmbeddings
from flair.data import Sentence
from flair.embeddings import DocumentPoolEmbeddings
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def extractQA():
    dataset = []
    with open("../Labs/Lab7/S10_question_answer_pairs.txt") as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if i != 0:
                question = line.split("\t")[1]
                answer = line.split("\t")[2]
                if question != "NULL" and answer != "NULL":
                    qa_row = {question: answer}  # TODO przemysl czy nie chcesz tego zapakowac do jednego duzego slownika
                    dataset.append(qa_row)
    return dataset


# TODO 1. Zwektoryzuj pytanie uzytkownika oraz pytania w bazie

# TODO 2. Podaj najbardziej podobny rekord z bazy na podstawie wzoru
def main():
    dataset = extractQA()

    embeddings = FlairEmbeddings('pl-forward')

    sentences = Sentence("Was allessandro volta an professor?")  # TODO NA KONCU -> niech user zada pytanie
    embeddings.embed(sentences)

    sentence_embeddings = DocumentPoolEmbeddings([FlairEmbeddings('pl-forward')])
    sentence_embeddings.embed(sentences)

    tfIdfVectorizer = TfidfVectorizer(use_idf=True)
    tfIdf = tfIdfVectorizer.fit_transform(dataset[0])  # dataset
    tfIdf[0].todense()

    df = pd.DataFrame(tfIdf[0].T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=["TF-IDF"])
    df = df.sort_values('TF-IDF', ascending=False)
    print(df.head(25))

    gwv = KeyedVectors.load_word2vec_format("../Labs/Lab7/glove_100_3_polish.txt")

    tokens = []
    labels = []

    words = list(gwv.index_to_key)
    for word in words[:1000]:
        tokens.append(gwv[word])
        labels.append(word)

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()
