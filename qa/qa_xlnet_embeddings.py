import torch, json, sys
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../paragraph_ranking')
from utils import get_xlnet_embeddings

print("Loading data...")

with open('../data/ranking/q{}_train.json'.format(sys.argv[1])) as train_file:
    train_set = json.load(train_file)

with open('../data/ranking/q{}_dev.json'.format(sys.argv[1])) as dev_file:
    dev_set = json.load(dev_file)

with open('../data/ranking/q{}_test.json'.format(sys.argv[1])) as test_file:
    test_set = json.load(test_file)

print("Data loading completed.")

dev_len = len(dev_set)
train_set += dev_set[:(2*dev_len)//3]
test_set += dev_set[(2*dev_len)//3:]

print("Extracting features...")

X_train = np.zeros((len(train_set), 768))
y_train = np.zeros(len(train_set))
for i, a in enumerate(train_set):
    article = a['article']
    X_train[i] = get_xlnet_embeddings(article).detach().cpu().numpy()
    y_train[i] = a['answer']

X_test = np.zeros((len(test_set), 768))
y_test = np.zeros(len(test_set))
for i, a in enumerate(test_set):
    article = a['article']
    X_test[i] = get_xlnet_embeddings(article).detach().cpu().numpy()
    y_test[i] = a['answer']

print("Feature extraction completed.")

print("Fitting model...")
model = LogisticRegression()
model.fit(X_train, y_train)
print("Model fitted.")

predicted = model.predict(X_test)
print("Accuracy: {}".format(accuracy_score(y_test, predicted)))
print("F1 Macro: {}".format(f1_score(y_test, predicted, average='macro')))
