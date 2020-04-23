import json, statistics

import numpy as np
# from gpt2_generation import generated_text_is_meaningful, get_generation_prefix
from rouge import Rouge
from scipy.stats import pearsonr

rouge = Rouge()
scores = []

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

def almost_the_same(a, b):
    len_ratio = len(a) / len(b) if len(a) < len(b) else len(b) / len(a)
    similarity = fuzz.partial_ratio(a, b)
    return len_ratio >= 0.9 and similarity > 90


def generated_text_is_meaningful(text, generation_prefix):
    return text != '' and not text.isspace() and not almost_the_same(text, generation_prefix)

def get_generation_prefix(article, question):
    generation_prefix = '<|startoftext|>' + '\n'
    generation_prefix += article + '\n'
    generation_prefix += 'QUESTION: ' + question + '\n'
    generation_prefix += 'EXPLANATION: '
    return generation_prefix


def evaluate_with_explanation():
    for i in range(1, 11):
        with open('../data/ttt/q{}_test.json'.format(i)) as test_file:
            articles = json.load(test_file)

        articles = [article for article in articles if
                    (i != 5 and article['answer'] == 1) or (i == 5 and article['answer'] == 0)]
        failure_rate = 0
        for article in articles:
            generation_prefix = get_generation_prefix(article['article'], article['question'])
            if not generated_text_is_meaningful(article['explanation_gpt2_sep_sat'], generation_prefix):
                failure_rate += 1
            try:
                scores.append(rouge.get_scores(article['explanation_gpt2_sep_sat'], article['explanation']))
            except:
                print('bad scrapping :/')
            # try:
            #     scores.append(rouge.get_scores(article['explanation_textrank'], article['explanation']))
            # except:
            #     print('bad scrapping :/')

        print('Q{} stats:'.format(i))
        failure_rate /= len(articles)
        print('Failure Rate is: {}'.format(failure_rate))

        rouge_1 = [[score[0]['rouge-1']['f'], score[0]['rouge-1']['p'], score[0]['rouge-1']['r']] for score in scores]
        print('ROUGE 1 (f, p, r): {}'.format(np.mean(rouge_1, axis=0)))
        rouge_2 = [[score[0]['rouge-2']['f'], score[0]['rouge-2']['p'], score[0]['rouge-2']['r']] for score in scores]
        print('ROUGE 2 (f, p, r): {}'.format(np.mean(rouge_2, axis=0)))
        rouge_l = [[score[0]['rouge-l']['f'], score[0]['rouge-l']['p'], score[0]['rouge-l']['r']] for score in scores]
        print('ROUGE L (f, p, r): {}'.format(np.mean(rouge_l, axis=0)))
        print('#######################################################################')


def evaluate_with_annotated_data():
    for i in range(1, 10):
        with open('../data/ttt/q{}_test.json'.format(i)) as test_file:
            articles = json.load(test_file)

        articles = [article for article in articles if
                    (i != 5 and article['answer'] == 1) or (i == 5 and article['answer'] == 0)]

        with open('../data/annotated/q{}_test.json'.format(i)) as test_file:
            annotated_articles = json.load(test_file)

        annotated_articles = [a for a in annotated_articles if len(a['annotations']) > 0]

        failure_rate = 0
        for article in annotated_articles:
            other_article = None
            for _article in articles:
                if article['article'] == _article['article']:
                    other_article = _article
                    break

            generation_prefix = get_generation_prefix(article['article'], article['question'])
            if not generated_text_is_meaningful(other_article['explanation_gpt2_sep_sat'], generation_prefix):
                failure_rate += 1

            sorted(article['annotations'])
            reference = ' '.join([article['sentences'][id] for id in article['annotations']])
            try:
                scores.append(rouge.get_scores(other_article['explanation_gpt2_sep_sat'], reference))
            except:
                print('bad scrapping :/')

        print('Q{} stats:'.format(i))
        failure_rate /= len(annotated_articles)
        print('Failure Rate is: {}'.format(failure_rate))

        rouge_1 = [[score[0]['rouge-1']['f'], score[0]['rouge-1']['p'], score[0]['rouge-1']['r']] for score in scores]
        print('ROUGE 1 (f, p, r): {}'.format(np.mean(rouge_1, axis=0)))
        rouge_2 = [[score[0]['rouge-2']['f'], score[0]['rouge-2']['p'], score[0]['rouge-2']['r']] for score in scores]
        print('ROUGE 2 (f, p, r): {}'.format(np.mean(rouge_2, axis=0)))
        rouge_l = [[score[0]['rouge-l']['f'], score[0]['rouge-l']['p'], score[0]['rouge-l']['r']] for score in scores]
        print('ROUGE L (f, p, r): {}'.format(np.mean(rouge_l, axis=0)))
        print('#######################################################################')


def measure_pearsonr_for_annotation_agreement():
    i = 1
    with open('../data/annotated/q{}_train.json'.format(i)) as test_file:
        annotated_articles = json.load(test_file)

    with open('../data/annotated/q{}_test.json'.format(i)) as test_file:
        annotated_articles += json.load(test_file)

    doubly_annotated_articles = []
    for article in annotated_articles:
        annotation_info = article['annotations_orig']
        annotators = set()
        for annotation in annotation_info:
            annotators.add(annotation['annotator'])

        if len(annotators) == 2:
            article['annotators'] = list(annotators)
            doubly_annotated_articles.append(article)

    data1 = []
    data2 = []
    for article in doubly_annotated_articles:
        count1 = len([a for a in article['annotations_orig'] if a['annotator'] == article['annotators'][0]])
        data1.append(count1)
        count2 = len([a for a in article['annotations_orig'] if a['annotator'] == article['annotators'][1]])
        data2.append(count2)

    correlation, _ = pearsonr(data1, data2)

    return correlation


def measure_annotator_match_rate():
    i = 1
    with open('../data/annotated/q{}_train.json'.format(i)) as test_file:
        annotated_articles = json.load(test_file)

    with open('../data/annotated/q{}_test.json'.format(i)) as test_file:
        annotated_articles += json.load(test_file)

    doubly_annotated_articles = []
    for article in annotated_articles:
        annotation_info = article['annotations_orig']
        annotators = set()
        for annotation in annotation_info:
            annotators.add(annotation['annotator'])

        if len(annotators) == 2:
            article['annotators'] = list(annotators)
            doubly_annotated_articles.append(article)

    match_rates = []
    for article in doubly_annotated_articles:
        annotations1 = set([a['sentenceIndex'] for a in article['annotations_orig'] if a['annotator'] == article['annotators'][0]])
        annotations2 = set([a['sentenceIndex'] for a in article['annotations_orig'] if a['annotator'] == article['annotators'][1]])

        match_count = len(annotations1.intersection(annotations2))
        total_count = len(annotations1.union(annotations2))

        match_rates.append(match_count/total_count)

    print(match_rates)
    return statistics.mean(match_rates), statistics.stdev(match_rates)


if __name__ == "__main__":
    print(measure_annotator_match_rate())
