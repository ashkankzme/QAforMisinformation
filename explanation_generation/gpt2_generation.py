import json
import sys

import gpt_2_simple as gpt2

sys.path.insert(1, '../paragraph_ranking')
from rank_gold_standard import biased_textrank


def select_top_k_texts_preserving_order(texts, ranking, k):
    texts_sorted = sorted(zip(texts, ranking), key=lambda item: item[1], reverse=True)
    top_texts = texts_sorted[:k]
    top_texts = [t[0] for t in top_texts]
    result = []
    for text in texts:
        if text in top_texts:
            result.append(text)
    return result


def generate_explanation(article, question, session):
    generation_prefix = '<|startoftext|>' + '\n'
    generation_prefix += article + '\n'
    generation_prefix += 'QUESTION: ' + question + '\n'
    generation_prefix += 'EXPLANATION: '
    return gpt2.generate(session, prefix=generation_prefix, truncate='<|endoftext|>', length=300, include_prefix=False,
                         temperature=0.7)


MODEL_NAME = '124M'
TRAINING_DATA_PATH = '../data/generation_input/train.txt'

session = gpt2.start_tf_sess()
gpt2.finetune(session, TRAINING_DATA_PATH, model_name=MODEL_NAME, steps=1000)

data_points_summarized = 0
for file_number in range(1, 11):
    print('processing file {} training data...'.format(file_number))
    with open('../data/qa_input_no_validation/q{}_train.json'.format(file_number)) as train_file:
        articles = json.load(train_file)

    for article in articles:
        try:
            article['generated_explanation'] = generate_explanation(article['article'], article['question'], session)
        except:
            data_points_summarized += 1
            ranking, texts = biased_textrank(article['article'], article['question'], 'OK', damping_factor=0.5)
            top_sentences = select_top_k_texts_preserving_order(texts, ranking, 30)
            article_summary = ' '.join(top_sentences)
            article['generated_explanation'] = generate_explanation(article_summary, article['question'], session)

    with open('../data/ranking/q{}_train.json'.format(file_number), 'w') as f:
        f.write(json.dumps(articles))

    print('results for training data of file {} saved. Data points summarized so far: {}'.format(file_number,
                                                                                                 data_points_summarized))

    print('processing file {} test data...'.format(file_number))
    with open('../data/qa_input_no_validation/q{}_test.json'.format(file_number)) as test_file:
        articles = json.load(test_file)

    for article in articles:
        try:
            article['generated_explanation'] = generate_explanation(article['article'], article['question'], session)
        except:
            data_points_summarized += 1
            ranking, texts = biased_textrank(article['article'], article['question'], 'OK', damping_factor=0.5)
            top_sentences = select_top_k_texts_preserving_order(texts, ranking, 30)
            article_summary = ' '.join(top_sentences)
            article['generated_explanation'] = generate_explanation(article_summary, article['question'], session)

    with open('../data/ranking/q{}_test.json'.format(file_number), 'w') as f:
        f.write(json.dumps(articles))

    print('results for test data of file {} saved. Data points summarized so far: {}'.format(file_number,
                                                                                             data_points_summarized))

print('all explanations generated, total summarized are: {}.'.format(data_points_summarized))
