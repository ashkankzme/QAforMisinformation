import json
import math
from random import sample

import numpy as np
from utils import get_paragraphs, get_bert_embeddings, cosine_similarity


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


# takes as input a paragraph, plus a set of
# paragraphs and their ranking return a
# weighted average of the rankings, with the
# weights being the cosine similarity of
# the given paragraph and the ranked paragraphs.
def get_relative_ranking(paragraph, source_paragraphs_embeddings, rankings):
    paragraph_embedding = get_bert_embeddings(paragraph)

    result = 0
    for p_embedding, ranking in zip(source_paragraphs_embeddings, rankings):
        result += cosine_similarity(paragraph_embedding, p_embedding) * ranking

    return result / len(rankings)


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

    NUMBER_OF_TOP_PARAGRAPHS_TO_INCLUDE = 3
    training_rankings = []
    for article in train:
        exp_similarities, _, paragraphs_embeddings, paragraphs = get_paragraph_similarities(article['article'],
                                                                                            article['question'],
                                                                                            article['explanation'])

        training_rankings.append([paragraphs_embeddings, exp_similarities])
        sorted_rankings = [x for _, x in sorted(zip(exp_similarities, paragraphs), key=lambda pair: pair[0])]

        new_text = ''
        for i in range(NUMBER_OF_TOP_PARAGRAPHS_TO_INCLUDE):
            new_text += sorted_rankings[-i] + '\n'

        article['article'] = new_text

    for article in test:
        sampled_training_instances = sample(training_rankings, math.floor(0.1 * len(training_rankings)))
        paragraphs = get_paragraphs(article['article'])

        paragraph_rankings = []

        for p in paragraphs:
            ranking = 0
            for training_paragraph in sampled_training_instances:
                ranking += get_relative_ranking(p, training_paragraph[0], training_paragraph[1])
            ranking /= len(sampled_training_instances)
            paragraph_rankings.append([ranking, p])

        sorted_rankings = [x for _, x in sorted(paragraph_rankings, key=lambda pair: pair[0])]
        new_text = ''
        print('pooo' + str(len(sorted_rankings)))
        for i in range(NUMBER_OF_TOP_PARAGRAPHS_TO_INCLUDE):
            new_text += sorted_rankings[-i] + '\n'

        article['article'] = new_text

    dev, test = test[:math.floor(len(test)/2)], test[math.floor(len(test)/2):]

    with open('../data/question_answering/q{}_train.json'.format(qid), 'w') as f:
        f.write(json.dumps(train))

    with open('../data/question_answering/q{}_dev.json'.format(qid), 'w') as f:
        f.write(json.dumps(dev))

    with open('../data/question_answering/q{}_test.json'.format(qid), 'w') as f:
        f.write(json.dumps(test))


def main():
    for i in range(1, 11):
        rank_train_data_for_question(i)

if __name__ == "__main__":
    main()
