import json
import math
import random
import sys
import numpy as np

sys.path.insert(1, '../paragraph_ranking')
from utils import get_paragraphs, get_bert_marked_text, tokenizer

with open('../data/news.json') as news_file:
    news = json.load(news_file)

with open('../data/stories.json') as story_file:
    stories = json.load(story_file)

articles = news + stories

print(str(len(articles)) + ' articles loaded.')

news_criteria = [c['question'] for c in articles[0]['criteria']]
story_criteria = [c['question'] for c in articles[2000]['criteria']]

to_be_deleted = []
# this is for removing duplicate articles
# there were a few articles that had
# page not found as their original text
# and we had to remove them.
# we also remove unnecessarily long articles
original_articles_map = {}
for i, _article in enumerate(articles):
    _criteria = _article['criteria'] if 'criteria' in _article else []
    paragraphs = get_paragraphs(_article['original_article']) if 'original_article' in _article else []

    if 'rating' not in _article or _article['rating'] == -1 or 'criteria' not in _article or len(
            _article['criteria']) < len(news_criteria) or 'original_article' not in _article or _article[
        'original_article'].isspace() or len(paragraphs) > 50 or len(paragraphs) <= 5 or len(
        [1 for p in paragraphs if len(tokenizer.tokenize(get_bert_marked_text(p))) > 512]) > 0 or \
            len([1 for q in _criteria if len(tokenizer.tokenize(get_bert_marked_text(q['explanation']))) > 512]) > 0 or \
            'Error code 404'.lower() in _article['original_article'].lower():
        to_be_deleted.append(i)
    elif _article['original_article'] not in original_articles_map:
        original_articles_map[_article['original_article']] = [i]
    else:
        original_articles_map[_article['original_article']].append(i)

duplicate_indices = [original_articles_map[duplicate_article] for duplicate_article in original_articles_map if
                     len(original_articles_map[duplicate_article]) > 1]
for index_list in duplicate_indices:
    for i in index_list:
        to_be_deleted.append(i)

doc_lens = []
for article in articles:
    if 'original_article' not in article:
        continue
    tokenized_article = tokenizer.tokenize(get_bert_marked_text(article['original_article']))
    doc_lens.append(len(tokenized_article))

avg_doc_len = np.mean(doc_lens)
print("Average doc len: {}".format(avg_doc_len))
std_doc_len = np.std(doc_lens)
print("Std doc len: {}".format(std_doc_len))

for i, article in enumerate(articles):
    if 'original_article' not in article:
        continue
    if doc_lens[i] > avg_doc_len + 2 * std_doc_len or doc_lens[i] < avg_doc_len - 2 * std_doc_len:
        to_be_deleted.append(i)

to_be_deleted = list(set(to_be_deleted))

for index in sorted(to_be_deleted, reverse=True):
    del articles[index]
    if index < len(news):
        del news[index]
    else:
        del stories[index - len(news)]

print('data cleaned. ' + str(len(articles)) + ' articles left. Count of story reviews: ' + str(
    len(stories)) + ', count of news reviews: ' + str(len(news)) + '.')

# extracting questions and their explanations
for i in range(10):
    qi = [{'article': article['original_article'], 'question': article['criteria'][i]['question'],
           'explanation': article['criteria'][i]['explanation'],
           'answer': 0 if article['criteria'][i]['answer'] == 'Not Satisfactory' else 1 if article['criteria'][i][
                                                                                               'answer'] == 'Satisfactory' else 2}
          for article in articles]
    # train/dev/test split: 70/15/15
    seed = i  # for getting the same random results every time
    random.Random(seed).shuffle(qi)

    split_index = math.floor(0.8 * len(qi))

    qi_train = qi[:split_index]
    qi_test = qi[split_index:]

    with open('../data/ttt/q{}_train.json'.format(i + 1), 'w') as f:
        f.write(json.dumps(qi_train))

    with open('../data/ttt/q{}_test.json'.format(i + 1), 'w') as f:
        f.write(json.dumps(qi_test))
