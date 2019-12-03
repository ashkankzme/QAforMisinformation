import json
import sys
import random

import numpy as np
from utils import get_paragraphs, get_sentences, get_bert_embeddings, cosine_similarity
from rank_gold_standard import biased_textrank, get_similarities
from sklearn.cluster import KMeans
import torch


def extract_paragraph_features(articles):
    for i, article in enumerate(articles):
        # if random.uniform(0, 1) < 0.1:
        print('processing {}th entry...'.format(i))
        article['paragraph_features'] = []
        texts, q, exp = article['paragraphs'], article['question'], article['explanation']
        damping_factor = 0.5
        exp_similarities, q_similarities, text_embeddings = get_similarities(texts, q, exp)
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
        exp_textrank = torch.ones((len(matrix), 1)) / len(matrix)
        iterations = 40
        for i in range(iterations):
            exp_textrank = torch.mm(scaled_matrix.t(), exp_textrank)
        bias = torch.tensor(q_similarities)
        scaled_matrix = damping_factor * matrix + (1 - damping_factor) * bias
        # scaled_matrix = s * matrix + (1 - s) / len(matrix)
        for row in scaled_matrix:
            row /= torch.sum(row)
        q_textrank = torch.ones((len(matrix), 1)) / len(matrix)
        iterations = 40
        for i in range(iterations):
            q_textrank = torch.mm(scaled_matrix.t(), exp_textrank)
        for i, paragraph in enumerate(article['paragraphs']):
            article['paragraph_features'].append(
                [exp_similarities[i], exp_textrank[i], q_similarities[i], q_textrank[i]])
    paragraph_features = []
    paragraphs = []
    for article in train_set:
        paragraph_features.extend(article['paragraph_features'])
        paragraphs.extend(article['paragraphs'])
    paragraph_features = np.array(paragraph_features)
    return paragraph_features, paragraphs

print("Loading data...")

with open('../data/qa_input_no_validation/q{}_train.json'.format(2)) as train_file:
    train_set = json.load(train_file)

with open('../data/qa_input_no_validation/q{}_test.json'.format(2)) as test_file:
    test_set = json.load(test_file)

print("Data loading completed.")

# parsing the articles into paragraphs
for article in train_set + test_set:
    article['paragraphs'] = get_paragraphs(article['article'])

# clustering for classifying paragraphs into relevant/irrelevant
print('Calculating paragraph features...')
paragraph_features_train, paragraphs_train = extract_paragraph_features(train_set)
print('Paragraph features calculated.')

# Doing some clustering
kmeans = KMeans(n_clusters=2, random_state=0).fit(paragraph_features_train)

paragraph_features_test, paragraphs_test = extract_paragraph_features(test_set)

predicted = kmeans.predict(paragraph_features_test)
print(len([1 for a in predicted if a == 1]))
print(len([1 for a in predicted if a == 0]))



# # initial classifier, based on string matching
# articles_with_referencing_explanations = []
#
# for article in train_set:
#     exp = article['explanation']
#     there_is_reference = exp.count('“') > 0 or exp.count('”') > 0 or exp.count('"') > 0
#     if there_is_reference:
#         articles_with_referencing_explanations.append(article)
#
# print(len(articles_with_referencing_explanations))
# print(len(train_set))
#
# for article in random.sample(articles_with_referencing_explanations, 10):
#     exp = article['explanation']
#
#     i = exp.find('“')
#     j = exp.find('”')
#
#     if i > j:
#         i, j = j, i
#
#     if i == -1 or j == -1:
#         print(i)
#         print(j)
#
#
#     extract = exp[i + 1:j]
#     for paragraph in article['paragraphs']:
#         if extract in paragraph:
#             print(extract)
#             print('##########################')
#             print(paragraph)
#             print('##########################')
#             print(exp)
#             print('##########################')


# # generating ground truth for paragraph relevance
# for article in train_set:
#     paragraph_labels = []
#     for paragraph in article['paragraphs']:
