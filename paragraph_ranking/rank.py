from utils import get_paragraphs, get_bert_embeddings, cosine_similarity
import numpy as np
import json


# gets as input a news article, a question and
# the explanation for the questions answer.
# ranks them by their similarities of their
# BERT embeddings to the explanation and the
# question.
def get_paragraph_similarities(text, q, exp):
    exp_embeddings = get_bert_embeddings(exp.strip())
    q_embeddings = get_bert_embeddings(q.strip())

    paragraphs = get_paragraphs(text)
    paragraphs_embeddings = [get_bert_embeddings(p) for p in paragraphs]

    # calculate cosine similarities between articles paragraphs and explanations or questions
    exp_similarities = [0] * len(paragraphs)
    q_similarities = [0] * len(paragraphs)
    for i, pe in enumerate(paragraphs_embeddings):
        exp_similarities[i] = cosine_similarity(pe, exp_embeddings)
        q_similarities[i] = cosine_similarity(pe, q_embeddings)

    return exp_similarities, q_similarities, paragraphs_embeddings, paragraphs


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
    similarities, paragraphs = get_paragraph_similarities(text, q, exp)

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


# takes as input a question id, which
# is the number of the question its
# trying to rank data on. We take the
# cosine similarities of BERT embeddings
# of paragraphs to the explanation and
# the questions separately. We then train
# a simple model that predicts the
# Writes the rankings of test data
def rank_train_data_for_question(qid):

    with open('../data/ranking/q{}_train.json'.format(qid)) as train_file:
        train = json.load(train_file)

    with open('../data/ranking/q{}_test.json'.format(qid)) as test_file:
        test = json.load(test_file)

    with open('../data/ranking/q{}_dev.json'.format(qid)) as dev_file:
        test += json.load(dev_file)


