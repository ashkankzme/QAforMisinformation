import json
import random


def prepare_for_generation_all_together():
    print("Loading data...")

    train_set = []
    for i in range(1, 11):
        with open('../data/ttt/q{}_train.json'.format(i)) as train_file:
            train_set += json.load(train_file)

    print("Data loading completed.")

    random.Random(2017).shuffle(train_set)

    train_str = generate_training_string(train_set)

    with open('../data/generation_input/train.txt', 'w') as f:
        f.write(train_str)


def generate_training_string(articles):
    train_str = ''
    for article in articles:
        train_str += '<|startoftext|>' + '\n'
        train_str += article['article'] + '\n'
        train_str += 'QUESTION: ' + article['question'] + '\n'
        train_str += 'EXPLANATION: ' + article['explanation'] + '\n'
        train_str += '<|endoftext|>' + '\n'
    return train_str


def prepare_for_generation_per_question_satisfactory():
    for i in range(1, 11):
        print("Loading data...")

        with open('../data/ttt/q{}_train.json'.format(i)) as train_file:
            train_set = json.load(train_file)

        print("Data loading completed.")

        train_set_sat = [article for article in train_set if article['answer'] == 1]
        random.Random(2017).shuffle(train_set_sat)

        train_str = generate_training_string(train_set_sat)

        with open('../data/generation_input/q{}_sat.txt'.format(i), 'w') as f:
            f.write(train_str)

        print('Training data for question{} saved.'.format(i))


if __name__ == "__main__":
    prepare_for_generation_per_question_satisfactory()
