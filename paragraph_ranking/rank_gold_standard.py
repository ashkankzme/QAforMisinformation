import json
import math
from random import sample

import numpy as np
import torch
from utils import get_paragraphs, get_sentences, get_bert_embeddings, cosine_similarity, get_xlnet_embeddings


# gets as input a news article, a question and
# the explanation for the questions answer.
# ranks them by their similarities of their
# XLNet or BERT embeddings to the explanation
# and the question.
def get_similarities(text, q, exp):
    exp_embeddings = get_bert_embeddings(exp.strip())
    q_embeddings = get_bert_embeddings(q.strip())

    texts = get_sentences(text)
    texts_embeddings = [get_bert_embeddings(p) for p in texts]

    # calculate cosine similarities between articles paragraphs and explanations or questions
    exp_similarities = [0] * len(texts)
    q_similarities = [0] * len(texts)
    for i, pe in enumerate(texts_embeddings):
        exp_similarities[i] = cosine_similarity(pe, exp_embeddings)
        q_similarities[i] = cosine_similarity(pe, q_embeddings)

    return exp_similarities, q_similarities, texts_embeddings, texts


# takes as input a piece of text, plus a set of
# paragraphs and their ranking return a
# weighted average of the rankings, with the
# weights being the cosine similarity of
# the given piece of text and the ranked paragraphs.
def get_relative_ranking(text, source_paragraphs_embeddings, rankings):
    text_embedding = get_bert_embeddings(text)

    result = 0
    for p_embedding, ranking in zip(source_paragraphs_embeddings, rankings):
        result += cosine_similarity(text_embedding, p_embedding) * ranking

    return result / len(rankings)


# gets as input a news article, a question and
# the explanation for the questions answer.
# breaks the article into chunks and
# creates a graph representation of them
# using BERT or XLNet embeddings and cosine similarity
# to calculate edge weights of the graph.
# runs the textrank algorithm on it and returns
# the ranking of each text as a score,
# alongside the text itself.
def biased_textrank(text, q, exp, damping_factor=0.5):
    exp_similarities, q_similarities, text_embeddings, texts  = get_similarities(text, q, exp)

    text_similarities = {}
    for i, text in enumerate(texts):
        similarities = {}
        for j, embedding in enumerate(text_embeddings):
            if i != j:
                similarities[texts[j]] = cosine_similarity(embedding, text_embeddings[i])

        text_similarities[text] = similarities

    # create text rank matrix
    matrix = torch.zeros((len(texts), len(texts)))
    for i, i_text in enumerate(texts):
        for j, j_text in enumerate(texts):
            if i != j:
                matrix[i][j] = text_similarities[i_text][j_text]

    bias = torch.tensor(exp_similarities)
    scaled_matrix = damping_factor * matrix + (1 - damping_factor) * bias
    # scaled_matrix = s * matrix + (1 - s) / len(matrix)
    for row in scaled_matrix:
        row /= torch.sum(row)
    v = torch.ones((len(matrix), 1)) / len(matrix)
    iterations = 40
    for i in range(iterations):
        v = torch.mm(scaled_matrix.t(), v)
    return v, texts


# takes as input a question id, which
# is the number of the question its
# trying to rank data on. We take the
# cosine similarities of BERT embeddings
# of paragraphs to the explanation and
# the questions separately. We then train
# a simple model that predicts the
# Writes the rankings of test data
def prepare_data_for_qa_textrank(qid):
    with open('../data/ranking/q{}_train.json'.format(qid)) as train_file:
        train = json.load(train_file)

    with open('../data/ranking/q{}_test.json'.format(qid)) as test_file:
        test = json.load(test_file)

    with open('../data/ranking/q{}_dev.json'.format(qid)) as dev_file:
        dev = json.load(dev_file)

    NUMBER_OF_TOP_ITEMS_TO_INCLUDE = 5
    for article in train + dev + test:
        ranks, texts = biased_textrank(article['article'], article['question'], article['explanation'])

        if len(texts) > NUMBER_OF_TOP_ITEMS_TO_INCLUDE:
            sorted_rankings = [x for _, x in sorted(zip(ranks, texts), key=lambda pair: pair[0])]

            new_text = ''
            for i in range(NUMBER_OF_TOP_ITEMS_TO_INCLUDE):
                new_text += sorted_rankings[-i] + '\n'

            article['article'] = new_text

    with open('../data/question_answering_gold_standard_fine_grained_textrank_50/q{}_train.json'.format(qid), 'w') as f:
        f.write(json.dumps(train))

    with open('../data/question_answering_gold_standard_fine_grained_textrank_50/q{}_dev.json'.format(qid), 'w') as f:
        f.write(json.dumps(dev))

    with open('../data/question_answering_gold_standard_fine_grained_textrank_50/q{}_test.json'.format(qid), 'w') as f:
        f.write(json.dumps(test))


# takes as input a question id, which
# is the number of the question its
# trying to rank data on. We take the
# cosine similarities of BERT embeddings
# of paragraphs to the explanation and
# the questions separately. We then train
# a simple model that predicts the
# Writes the rankings of test data
def prepare_data_for_qa(qid):
    with open('../data/ranking/q{}_train.json'.format(qid)) as train_file:
        train = json.load(train_file)

    with open('../data/ranking/q{}_test.json'.format(qid)) as test_file:
        test = json.load(test_file)

    with open('../data/ranking/q{}_dev.json'.format(qid)) as dev_file:
        dev = json.load(dev_file)

    NUMBER_OF_TOP_PARAGRAPHS_TO_INCLUDE = 5
    for article in train + dev + test:
        exp_similarities, _, paragraphs_embeddings, paragraphs = get_similarities(article['article'],
                                                                                  article['question'],
                                                                                  article['explanation'])

        if len(paragraphs) > NUMBER_OF_TOP_PARAGRAPHS_TO_INCLUDE:
            sorted_rankings = [x for _, x in sorted(zip(exp_similarities, paragraphs), key=lambda pair: pair[0])]

            new_text = ''
            for i in range(NUMBER_OF_TOP_PARAGRAPHS_TO_INCLUDE):
                new_text += sorted_rankings[-i] + '\n'

            article['article'] = new_text

    with open('../data/question_answering_gold_standard_fine_grained_bert/q{}_train.json'.format(qid), 'w') as f:
        f.write(json.dumps(train))

    with open('../data/question_answering_gold_standard_fine_grained_bert/q{}_dev.json'.format(qid), 'w') as f:
        f.write(json.dumps(dev))

    with open('../data/question_answering_gold_standard_fine_grained_bert/q{}_test.json'.format(qid), 'w') as f:
        f.write(json.dumps(test))


def main():
    for i in range(8, 11):
        prepare_data_for_qa_textrank(i)

if __name__ == "__main__":
    main()
