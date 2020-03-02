import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, XLNetModel, XLNetTokenizer
import logging
import nltk


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
# tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
# model = XLNetModel.from_pretrained('xlnet-base-cased')
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)
model.eval()
model.to(device)


# gets as input two numpy vectors
# and returns the cosine similarity
# between them.
def cosine_similarity(a, b):
    return torch.abs(torch.dot(a, b) / (torch.norm(a) * torch.norm(b)))


# returns the marked text for BERT
# use, using the one sentence model
def get_bert_marked_text(a):
    return '[CLS] ' + a + ' [SEP]'

# returns the marked text for XLNet
# use, using the one sentence model
def get_xlnet_marked_text(a):
    return a + " [SEP] [CLS]"


# returns BERT embeddings of [CLS]
# (the last layer)
# for the input text.
def get_bert_embeddings(a):
    marked_text = get_bert_marked_text(a)
    tokenized_text = tokenizer.tokenize(marked_text)
    if len(tokenized_text) > 512:
        print("Warning: Long sequence input for BERT. Truncating anything larger than 512th token. Actual size: {}".format(len(tokenized_text)))
        tokenized_text = tokenized_text[:511] + tokenizer.tokenize('[SEP]')
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segment_ids = [1] * len(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segment_ids])
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)
    return encoded_layers[-1][0]  # the last layer of first batch for first token, [CLS]


# returns XLNet embeddings of [CLS]
# (the last layer)
# for the input text.
def get_xlnet_embeddings(a):
    marked_text = get_xlnet_marked_text(a)
    input_ids = torch.tensor(tokenizer.encode(marked_text)).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_ids)

    return outputs[0][0][-1]  # the last layer of first batch for the last token, [CLS]


# gets as input a body of text, including
# \n separated paragraphs. Outputs those
# paragraphs in a list of strings.
def get_paragraphs(text):
    paragraphs = text.split('\n')
    paragraphs = [p for p in paragraphs if p and not p.isspace()]
    return paragraphs


# gets as input a body of text, including
# \n separated paragraphs. Outputs sentences
# detected in text in a list of strings.
def get_sentences(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [s for s in sentences if s and not s.isspace()]
    return sentences
