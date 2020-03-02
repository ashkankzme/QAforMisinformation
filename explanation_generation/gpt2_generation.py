import json
import random
import sys

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

import gpt_2_simple as gpt2
import tensorflow as tf

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


def almost_the_same(a, b):
    len_ratio = len(a) / len(b) if len(a) < len(b) else len(b) / len(a)
    similarity = fuzz.partial_ratio(a, b)
    return len_ratio >= 0.9 and similarity > 90


def generated_text_is_meaningful(text, generation_prefix):
    return text != '' and not text.isspace() and not almost_the_same(text, generation_prefix)


def generate_explanation(article, question, session):
    generation_prefix = get_generation_prefix(article, question)
    temperature = 0.7
    while True:
        generated_explanations = gpt2.generate(session, prefix=generation_prefix, truncate='<|endoftext|>', length=80,
                                               include_prefix=False, temperature=temperature, return_as_list=True, batch_size=8,
                                               nsamples=8)
        for generated_explanation in generated_explanations:
            if generated_text_is_meaningful(generated_explanation, generation_prefix) or temperature >= 1:
                print(generated_explanation)
                return generated_explanation

        temperature += 0.1
        print("GPT-2 generated giberish, increasing temperature to {}".format(temperature))


def get_generation_prefix(article, question):
    generation_prefix = '<|startoftext|>' + '\n'
    generation_prefix += article + '\n'
    generation_prefix += 'QUESTION: ' + question + '\n'
    generation_prefix += 'EXPLANATION: '
    return generation_prefix


MODEL_NAME = '355M'
TRAINING_DATA_PATH = '../data/generation_input/train.txt'

session = gpt2.start_tf_sess()
# gpt2.finetune(session, TRAINING_DATA_PATH, model_name=MODEL_NAME, steps=1000)
gpt2.load_gpt2(session)

data_points_summarized = 0
for file_number in range(1, 11):
    print('processing file {} test data...'.format(file_number))
    with open('../data/ttt/q{}_test.json'.format(file_number)) as test_file:
        articles = json.load(test_file)
    for article in articles:
        # if random.uniform(0, 1) < 0.1:  # bug fix for slow down in generation
        #     tf.reset_default_graph()
        if 'explanation_gpt2' in article and generated_text_is_meaningful(article['explanation_gpt2'],
                                                                          get_generation_prefix(article['article'],
                                                                                                article['question'])):
            continue

        summary_size = 25
        summary_doesnt_fit = True
        article_text = article['article']
        while summary_doesnt_fit:
            try:
                article['explanation_gpt2'] = generate_explanation(article_text, article['question'], session)
                if generated_text_is_meaningful(article['explanation_gpt2'], get_generation_prefix(article_text, article['question'])):
                    summary_doesnt_fit = False
                else:
                    raise ValueError('Generated explanation was gibberish (whitespace or repeating precondition text)')
            except:
                if summary_size == 25:  # gotta make sure we only increment this once per article at most
                    data_points_summarized += 1
                ranking, texts = biased_textrank(article['article'], article['question'])
                top_sentences = select_top_k_texts_preserving_order(texts, ranking, summary_size)
                article_summary = ' '.join(top_sentences)
                article_text = article_summary
                summary_size -= 5

        if random.uniform(0, 1) < 0.1:
            with open('../data/ttt/q{}_test.json'.format(file_number), 'w') as f:
                f.write(json.dumps(articles))
            print('results for test data of file {} saved. Data points summarized so far: {}'.format(file_number,
                                                                                                     data_points_summarized))

    with open('../data/ttt/q{}_test.json'.format(file_number), 'w') as f:
        f.write(json.dumps(articles))
    print('results for test data of file {} saved. Data points summarized so far: {}'.format(file_number,
                                                                                             data_points_summarized))
    # print('processing file {} training data...'.format(file_number))
    # with open('../data/qa_input_no_validation/q{}_train.json'.format(file_number)) as train_file:
    #     articles = json.load(train_file)
    # for article in articles:
    #     summary_size = 30
    #     summary_doesnt_fit = True
    #     article_text = article['article']
    #     while summary_doesnt_fit:
    #         try:
    #             article['generated_explanation'] = generate_explanation(article_text, article['question'], session)
    #             summary_doesnt_fit = False
    #         except:
    #             if summary_size == 30:  # gotta make sure we only increment this once per article at most
    #                 data_points_summarized += 1
    #             ranking, texts = biased_textrank(article['article'], article['question'], 'OK', damping_factor=0.5)
    #             top_sentences = select_top_k_texts_preserving_order(texts, ranking, summary_size)
    #             article_summary = ' '.join(top_sentences)
    #             article_text = article_summary
    #             summary_size -= 5
    # with open('../data/ranking/q{}_train.json'.format(file_number), 'w') as f:
    #     f.write(json.dumps(articles))
    # print('results for training data of file {} saved. Data points summarized so far: {}'.format(file_number,
    #                                                                                              data_points_summarized))
print('all explanations generated, total summarized are: {}.'.format(data_points_summarized))
