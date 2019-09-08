import json
import math
import random

from paragraph_ranking import get_paragraphs

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
for i in range(len(articles)):
    _article = articles[i]
    if 'rating' not in _article or _article['rating'] == -1 or 'criteria' not in _article or len(
            _article['criteria']) < len(news_criteria) or 'original_article' not in _article or _article[
        'original_article'].isspace() or len(get_paragraphs(_article['original_article'])) > 50:
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

    first_split_idnex = math.floor(0.7 * len(qi))
    second_split_index = math.floor(0.85 * len(qi))

    qi_train = qi[:first_split_idnex]
    qi_dev = qi[first_split_idnex:second_split_index]
    qi_test = qi[second_split_index:]

    with open('../data/q{}_train.json'.format(i + 1), 'w') as f:
        f.write(json.dumps(qi_train))

    with open('../data/q{}_dev.json'.format(i + 1), 'w') as f:
        f.write(json.dumps(qi_dev))

    with open('../data/q{}_test.json'.format(i + 1), 'w') as f:
        f.write(json.dumps(qi_test))
