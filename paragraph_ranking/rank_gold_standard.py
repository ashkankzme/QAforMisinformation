import json
import math
import sys
from random import sample

import numpy as np
import torch
from utils import get_paragraphs, get_sentences, cosine_similarity, select_top_k_texts_preserving_order, get_bert_embeddings, get_xlnet_embeddings


# takes an array/matrix as input
# rescales everything to (0, 1)
# range, proportionally
def rescale(a):
    maximum = torch.max(a)
    minimum = torch.min(a)
    return (a - minimum) / (maximum - minimum)


# gets as input a news article, a question and
# the explanation for the questions answer.
# ranks them by their similarities of their
# XLNet or BERT embeddings to the explanation
# and the question.
def get_similarities(texts, q, exp):
    exp_embeddings = get_bert_embeddings(exp.strip())
    q_embeddings = get_bert_embeddings(q.strip())

    texts_embeddings = [get_bert_embeddings(p) for p in texts]

    # calculate cosine similarities between articles paragraphs and explanations or questions
    exp_similarities = [0] * len(texts)
    q_similarities = [0] * len(texts)
    for i, pe in enumerate(texts_embeddings):
        exp_similarities[i] = cosine_similarity(pe, exp_embeddings)
        q_similarities[i] = cosine_similarity(pe, q_embeddings)

    return exp_similarities, q_similarities, texts_embeddings


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
def biased_textrank(texts, bias_text, damping_factor=0.8, similarity_threshold=0.78):
    texts_embeddings = [get_bert_embeddings(p) for p in texts]

    text_similarities = {}
    for i, text in enumerate(texts):
        similarities = {}
        for j, embedding in enumerate(texts_embeddings):
            if i != j:
                similarities[texts[j]] = cosine_similarity(embedding, texts_embeddings[i])

        text_similarities[text] = similarities

    # create text rank matrix, add edges between pieces that are more than X similar
    matrix = torch.zeros((len(texts), len(texts)))
    for i, i_text in enumerate(texts):
        for j, j_text in enumerate(texts):
            if i != j and text_similarities[i_text][j_text] > similarity_threshold:
                matrix[i][j] = text_similarities[i_text][j_text]

    # matrix = rescale(matrix)

    # preparing to add bias
    bias_embedding = get_bert_embeddings(bias_text.strip())

    bias_text_similarities = torch.zeros(len(texts))
    for i, text_embedding in enumerate(texts_embeddings):
        bias_text_similarities[i] = cosine_similarity(text_embedding, bias_embedding)

    bias_text_similarities = rescale(bias_text_similarities)

    bias = torch.tensor(bias_text_similarities)
    scaled_matrix = damping_factor * matrix + (1 - damping_factor) * bias
    # scaled_matrix = damping_factor * matrix + (1 - damping_factor) / len(matrix)

    for row in scaled_matrix:
        row /= torch.sum(row)
    # scaled_matrix = rescale(scaled_matrix)

    print('Calculating ranks...')
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
        paragraphs = get_sentences(article['article'])
        exp_similarities, _, paragraphs_embeddings = get_similarities(paragraphs,
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


def extract_explanations_with_textrank(qid, summary_size):
    with open('../data/ttt/q{}_train.json'.format(qid)) as train_file:
        train_set = json.load(train_file)

    with open('../data/ttt/q{}_test.json'.format(qid)) as test_file:
        test_set = json.load(test_file)

    for article in train_set + test_set:
        article_text = article['article']
        article_sentences = get_sentences(article_text)
        question = article['question']

        ranking, _ = biased_textrank(article_sentences, question)
        top_sentences = select_top_k_texts_preserving_order(article_sentences, ranking, summary_size)

        article['explanation_textrank'] = ' '.join(top_sentences)

    with open('../data/ttt/q{}_train.json'.format(qid), 'w') as f:
        f.write(json.dumps(train_set))

    with open('../data/ttt/q{}_test.json'.format(qid), 'w') as f:
        f.write(json.dumps(test_set))


def extract_explanations_with_bert_embeddings(qid, summary_size):
    with open('../data/ttt/q{}_train.json'.format(qid)) as train_file:
        train_set = json.load(train_file)

    with open('../data/ttt/q{}_test.json'.format(qid)) as test_file:
        test_set = json.load(test_file)

    for article in train_set + test_set:
        article_text = article['article']
        article_sentences = get_sentences(article_text)
        sentence_embeddings = [get_bert_embeddings(sentence.strip()) for sentence in article_sentences]

        question = article['question']
        bias_embedding = get_bert_embeddings(question.strip())

        bias_text_similarities = torch.zeros(len(article_sentences))
        for i, sentence_embedding in enumerate(sentence_embeddings):
            bias_text_similarities[i] = cosine_similarity(sentence_embedding, bias_embedding)

        top_sentences = select_top_k_texts_preserving_order(article_sentences, bias_text_similarities, summary_size)
        article['explanation_bert_embeddings'] = ' '.join(top_sentences)

    with open('../data/ttt/q{}_train.json'.format(qid), 'w') as f:
        f.write(json.dumps(train_set))

    with open('../data/ttt/q{}_test.json'.format(qid), 'w') as f:
        f.write(json.dumps(test_set))


def main():
    begin = int(sys.argv[1])
    end = int(sys.argv[2])
    for i in range(begin, end):
        extract_explanations_with_textrank(i, 5)
        extract_explanations_with_bert_embeddings(i, 5)


if __name__ == "__main__":
    main()
