import json

for Q_NUMBER in range(1, 11):
    print("Loading data...")

    with open('../data/paragraph_relevance_classification_input/q{}_train.json'.format(Q_NUMBER)) as train_file:
        train_set = json.load(train_file)

    with open('../data/paragraph_relevance_classification_input/q{}_test.json'.format(Q_NUMBER)) as test_file:
        test_set = json.load(test_file)

    print("Data loading completed.")

    for entry in train_set + test_set:
        if 'paragraph_relevance_learned_labels' in entry:
            del entry['paragraph_relevance_learned_labels']

    with open('../data/paragraph_relevance_classification_input/q{}_train.json'.format(Q_NUMBER), 'w') as f:
        f.write(json.dumps(train_set))

    with open('../data/paragraph_relevance_classification_input/q{}_test.json'.format(Q_NUMBER), 'w') as f:
        f.write(json.dumps(test_set))