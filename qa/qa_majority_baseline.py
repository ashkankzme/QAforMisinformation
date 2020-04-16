import json

from sklearn.metrics import f1_score, accuracy_score


def get_majority_baseline(qid):
    print("Loading data...")
    with open('../data/ttt/q{}_test.json'.format(qid)) as test_file:
        test_set = json.load(test_file)
    print("Data loading completed.")

    labels_test = [1 if (qid != 5 and article['answer'] == 1) or (qid == 5 and article['answer'] == 0) else 0 for
                   article in test_set]
    ones = len([label for label in labels_test if label == 1])
    zeros = len([label for label in labels_test if label == 0])
    majority_class = 1 if ones >= zeros else 0
    predictions = len(labels_test) * [majority_class]

    print("Test Accuracy: {}".format(accuracy_score(labels_test, predictions)))
    print("F1 pos: {}".format(f1_score(labels_test, predictions, pos_label=1)))
    print("F1 neg: {}".format(f1_score(labels_test, predictions, pos_label=0)))


if __name__ == "__main__":
    for i in range(1, 11):
        get_majority_baseline(i)
