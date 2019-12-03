import json
import sys
import random

# import numpy as np
from utils import get_paragraphs, get_sentences, get_bert_embeddings
from rank_gold_standard import biased_textrank

print("Loading data...")

with open('../data/qa_input_no_validation/q{}_train.json'.format(1)) as train_file:
    train_set = json.load(train_file)

with open('../data/qa_input_no_validation/q{}_test.json'.format(1)) as test_file:
    test_set = json.load(test_file)

print("Data loading completed.")

# parsing the articles into paragraphs
for article in train_set + test_set:
    article['paragraphs'] = get_paragraphs(article['article'])

# clustering for classifying paragraphs into relevant/irrelevant
paragraphs = []
for article in train_set:
    paragraphs.extend()

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
