from textrank_utilities import get_paragraphs, get_bert_embeddings, cosine_similarity
import numpy as np


# gets as input a news article, a question and
# the explanation for the questions answer.
# ranks them by their similarities of their
# BERT embeddings to the explanation and the
# question.
def rank_paragraphs(text, q, exp):
    qexp = q + ' ' + exp.strip()
    qexp_embeddings = get_bert_embeddings(qexp)

    paragraphs = get_paragraphs(text)
    paragraphs_embeddings = [get_bert_embeddings(p) for p in paragraphs]

    # calculate cosine similarities between articles paragraphs and qexp pair
    similarities = np.zeros(len(paragraphs))
    for i, pe in enumerate(paragraphs_embeddings):
        similarities[i] = cosine_similarity(pe, qexp_embeddings)

    return similarities, paragraphs


# gets as input a news article, a question and
# the explanation for the questions answer.
# breaks the article in to paragraphs and
# creates a graph representation of them
# using BERT embeddings and cosine similarity
# to calculate edge weights of the graph.
# runs the textrank algorithm on it and returns
# the ranking of each paragraph as a score,
# alongside the paragraph itself.
def textrank(text, q, exp):
    similarities, paragraphs = rank_paragraphs(text, q, exp)

    # create text rank matrix
    matrix = np.zeros((len(paragraphs), len(paragraphs)))
    for i, i_similarity in enumerate(similarities):
        for j, j_similarity in enumerate(similarities):
            if i_similarity < j_similarity:
                matrix[i][j] = j_similarity - i_similarity

    s = 0.9
    v = np.ones(len(matrix)) / len(matrix)
    scaled_matrix = s * matrix + (1 - s) / len(matrix)
    iterations = 20
    for i in range(iterations):
        v = scaled_matrix.dot(v)
    return v, paragraphs
