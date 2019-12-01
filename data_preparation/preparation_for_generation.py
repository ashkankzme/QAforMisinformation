import json, random


print("Loading data...")

train_set = []
for i in range(1, 11):
    with open('../data/qa_input_no_validation/q{}_train.json'.format(i)) as train_file:
        train_set += json.load(train_file)

print("Data loading completed.")

random.Random(2017).shuffle(train_set)

train_str = ''

for article in train_set:
    train_str += '<|startoftext|>' + '\n'
    train_str += article['article'] + '\n'
    train_str += 'QUESTION: ' + article['question'] + '\n'
    train_str += 'EXPLANATION: ' + article['explanation'] + '\n'
    train_str += '<|endoftext|>' + '\n'


with open('../data/generation_input/train.txt', 'w') as f:
    f.write(train_str)