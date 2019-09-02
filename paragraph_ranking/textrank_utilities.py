import torch
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM
import logging

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()


# gets as input two numpy vectors
# and returns the cosine similarity
# between them.
def cosine_similarity(a, b):
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))


# returns the marked text for BERT
# use, using the one sentence model
def get_bert_marked_text(a):
    return '[CLS] ' + a + ' [SEP]'


# returns BERT embeddings of [CLS]
# (the average of last 4 layers)
# for the input text.
def get_bert_embeddings(a):
    marked_text = get_bert_marked_text(a)
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segment_ids = [1] * len(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segment_ids])
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)
    return encoded_layers[-1][0]  # the last layer of first batch for first token, [CLS]


# gets as input a body of text, including
# \n separated paragraphs. Outputs those
# paragraphs in a list of strings.
def get_paragraphs(text):
    paragraphs = text.split('\n')
    paragraphs = [p for p in paragraphs if p]
    return paragraphs
