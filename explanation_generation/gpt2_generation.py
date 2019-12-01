import json

import gpt_2_simple as gpt2


def generate_explanation(article, session):
    generation_prefix = '<|startoftext|>' + '\n'
    generation_prefix += article['article'] + '\n'
    generation_prefix += 'QUESTION: ' + article['question'] + '\n'
    generation_prefix += 'EXPLANATION: '
    return gpt2.generate(session, prefix=generation_prefix, truncate='<|endoftext|>', length=300, inclue_prefix=False,
                         temperature=0.7)


MODEL_NAME = '124M'
TRAINING_DATA_PATH = '../data/generation_input/train.txt'

session = gpt2.start_tf_sess()
gpt2.finetune(session, TRAINING_DATA_PATH, model_name=MODEL_NAME, steps=1000)

for file_number in range(1, 11):
    with open('../data/qa_input_no_validation/q{}_train.json'.format(file_number)) as train_file:
        articles = json.load(train_file)

    for article in articles:
        article['generated_explanation'] = generate_explanation(article, session)

    with open('../data/ranking/q{}_train.json'.format(file_number), 'w') as f:
        f.write(json.dumps(articles))

    with open('../data/qa_input_no_validation/q{}_test.json'.format(file_number)) as test_file:
        articles = json.load(test_file)

    for article in articles:
        article['generated_explanation'] = generate_explanation(article, session)

    with open('../data/ranking/q{}_test.json'.format(file_number), 'w') as f:
        f.write(json.dumps(articles))