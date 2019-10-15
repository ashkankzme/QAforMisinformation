import torch, json, io, sys
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score

from transformers import XLNetModel, XLNetTokenizer, XLNetForSequenceClassification
from transformers import AdamW

from tqdm import tqdm, trange
import numpy as np

from .paragraph_ranking import get_xlnet_embeddings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print("GPU count: " + n_gpu)

print("Loading data...")

with open('../data/question_answering_gold_standard_fine_grained_bert/q{}_train.json'.format(sys.argv[1])) as train_file:
    train_set = json.load(train_file)

with open('../data/question_answering_gold_standard_fine_grained_bert/q{}_dev.json'.format(sys.argv[1])) as dev_file:
    dev_set = json.load(dev_file)

with open('../data/question_answering_gold_standard_fine_grained_bert/q{}_test.json'.format(sys.argv[1])) as test_file:
    test_set = json.load(test_file)

print("Data loading completed.")

dev_len = len(dev_set)
train_set += dev_set[:(2*dev_len)//3]
test_set += dev_set[(2*dev_len)//3:]

print(get_xlnet_embeddings(train_set[0]['article']))