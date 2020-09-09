import json
import random
import sys

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

import gpt_2_simple as gpt2
import tensorflow as tf

sys.path.insert(1, '../paragraph_ranking')
from rank_gold_standard import biased_textrank
from utils import get_sentences

split = sys.argv[1]
range_begin = sys.argv[2]
range_end = sys.argv[3]

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
                                               include_prefix=False, temperature=temperature, return_as_list=True, batch_size=2,
                                               nsamples=2)
        for generated_explanation in generated_explanations:
            if generated_text_is_meaningful(generated_explanation, generation_prefix) or temperature >= 0.8:
                print(generated_explanation)
                return generated_explanation

        temperature += 0.1


def get_generation_prefix(article, question):
    generation_prefix = '<|startoftext|>' + '\n'
    generation_prefix += article + '\n'
    generation_prefix += 'QUESTION: ' + question + '\n'
    generation_prefix += 'EXPLANATION: '
    return generation_prefix


MODEL_NAME = '355M'
TRAINING_DATA_PATH = '../data/generation_input/textrank/q{}_sat.txt'

data_points_summarized = 0
for file_number in range(int(range_begin), int(range_end)):
    print('fine-tuning a gpt-2 model for file {} {} data...'.format(file_number, split))
    session = gpt2.start_tf_sess()
    gpt2.finetune(session, TRAINING_DATA_PATH.format(file_number), model_name=MODEL_NAME, steps=400, run_name='q{}_sat_textrank'.format(file_number))
    # gpt2.load_gpt2(session, run_name='q{}_sat_textrank'.format(file_number))
    # print('processing file {} {} data...'.format(file_number, split))
    # with open('../data/ttt/q{}_{}.json'.format(file_number, split)) as test_file:
    #     articles = json.load(test_file)
    # for article_id, article in enumerate(articles):
    #     article_text = article['explanation_textrank']
    #
    #     if 'explanation_gpt2_textrank_sep_sat' in article and generated_text_is_meaningful(article['explanation_gpt2_textrank_sep_sat'],
    #                                                                       get_generation_prefix(article_text,
    #                                                                                             article['question'])):
    #         print('Skipping article #{} because it already has a meaningful generated explanation.'.format(article_id))
    #         continue
    #     # elif (file_number != 5 and article['answer'] != 1) or (file_number == 5 and article['answer'] != 0):
    #     #     print('Skipping article #{} because it\'s not satisfactory for question{}.'.format(article_id, file_number))
    #     #     continue
    #
    #     summary_size = 4
    #     summary_doesnt_fit = True
    #     while summary_doesnt_fit:
    #         try:
    #             print('Generating explanation for article #{} ...'.format(article_id))
    #             article['explanation_gpt2_textrank_sep_sat'] = generate_explanation(article_text, article['question'], session)
    #             if generated_text_is_meaningful(article['explanation_gpt2_textrank_sep_sat'], get_generation_prefix(article_text, article['question'])):
    #                 summary_doesnt_fit = False
    #             elif summary_size < 3:
    #                 article['explanation_gpt2_textrank_sep_sat'] = article['explanation_textrank']
    #             else:
    #                 print('Generated explanation for article #{} was not meaningful.'.format(article_id))
    #                 raise ValueError('Generated explanation was gibberish (whitespace or repeating precondition text)')
    #         except Exception as e:
    #             print(e)
    #             if summary_size == 4:  # gotta make sure we only increment this once per article at most
    #                 data_points_summarized += 1
    #
    #             print('Running biased textrank for article #{} ...'.format(article_id))
    #             ranking, texts = biased_textrank(get_sentences(article['explanation_textrank']), article['question'])
    #             print('Biased textrank completed.')
    #             top_sentences = select_top_k_texts_preserving_order(texts, ranking, summary_size)
    #             article_summary = ' '.join(top_sentences)
    #             article_text = article_summary
    #             summary_size -= 1
    #
    #     with open('../data/ttt/q{}_{}.json'.format(file_number, split), 'w') as f:
    #         f.write(json.dumps(articles))
    #     print('results for {} data of file {} saved. Data points summarized so far: {}'.format(split, file_number, data_points_summarized))
    #
    #     # K.clear_session()
    #     if article_id % 20 == 0:  # bug fix for slow down in generation
    #         tf.reset_default_graph()
    #         session = gpt2.start_tf_sess()
    #         gpt2.load_gpt2(session, run_name='q{}_sat_textrank'.format(file_number))

    # tf.reset_default_graph()

    # with open('../data/ttt/q{}_{}.json'.format(file_number, split), 'w') as f:
    #     f.write(json.dumps(articles))

# print('all explanations generated, total summarized are: {}.'.format(data_points_summarized))
