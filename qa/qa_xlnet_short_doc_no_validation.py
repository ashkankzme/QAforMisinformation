import torch, json, io, sys
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score

from transformers import XLNetModel, XLNetTokenizer, XLNetForSequenceClassification
from transformers import AdamW

from tqdm import tqdm, trange
import numpy as np

file_number = sys.argv[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print("GPU name: " + torch.cuda.get_device_name(0))

print("Loading data...")

with open('../data/ttt/q{}_train.json'.format(sys.argv[1])) as train_file:
    train_set = json.load(train_file)

with open('../data/ttt/q{}_test.json'.format(sys.argv[1])) as test_file:
    test_set = json.load(test_file)

print("Data loading completed.")

# Create sentence and label lists
sentences_train = [article['explanation_gpt2_textrank_sep_sat'] + " [SEP] [CLS]" for article in train_set]
sentences_test = [article['explanation_gpt2_textrank_sep_sat'] + " [SEP] [CLS]" for article in test_set]

labels_train = [1 if (file_number != 5 and article['answer'] == 1) or (file_number == 5 and article['answer'] == 0) else 0 for article in train_set]
labels_test = [1 if (file_number != 5 and article['answer'] == 1) or (file_number == 5 and article['answer'] == 0) else 0 for article in test_set]

tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=False)

tokenized_texts_train = [tokenizer.tokenize(sent) for sent in sentences_train]
tokenized_texts_test = [tokenizer.tokenize(sent) for sent in sentences_test]
print ("Tokenize the first sentence:")
print (tokenized_texts_train[0])

# Set the maximum sequence length. The longest sequence in our training set is 47, but we'll leave room on the end anyway.
MAX_LEN = 128
average_len = 0
reduced_inputs = 0
for tokens in tokenized_texts_train + tokenized_texts_test:
    average_len += len(tokens)
    if len(tokens) > MAX_LEN:
        reduced_inputs += 1

average_len /= len(tokenized_texts_train + tokenized_texts_test)

print("reduced input is: {}".format(reduced_inputs))
print("average_len is: {}".format(average_len))

# Use the XLNet tokenizer to convert the tokens to their index numbers in the XLNet vocabulary
input_ids_train = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts_train]
input_ids_test = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts_test]

# Pad our input tokens
input_ids_train = pad_sequences(input_ids_train, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
input_ids_test = pad_sequences(input_ids_test, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# remembering how input ids map to original input for inference
input_ids_map = {}
for i, point in enumerate(input_ids_test):
    input_ids_map[tuple(point)] = i

# Create attention masks
attention_masks_train = []

# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids_train:
  seq_mask = [float(i>0) for i in seq]
  attention_masks_train.append(seq_mask)

# Create attention masks
attention_masks_test = []

# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids_test:
  seq_mask = [float(i>0) for i in seq]
  attention_masks_test.append(seq_mask)

# Use train_test_split to split our data into train and validation sets for training

train_inputs, train_labels = input_ids_train, labels_train
train_masks = attention_masks_train

# Convert all of our data into torch tensors, the required datatype for our model

train_inputs = torch.tensor(train_inputs)
train_labels = torch.tensor(train_labels)
train_masks = torch.tensor(train_masks)

# Select a batch size for training. For fine-tuning with XLNet, the authors recommend a batch size of 32, 48, or 128. We will use 32 here to avoid memory issues.
batch_size = 32

# Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop,
# with an iterator the entire dataset does not need to be loaded into memory

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Load XLNEtForSequenceClassification, the pretrained XLNet model with a single linear classification layer on top.

model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=3)
if n_gpu > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = torch.nn.DataParallel(model)
model.cuda()

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

# This variable contains all of the hyperparemeter information our training loop needs
optimizer = AdamW(optimizer_grouped_parameters,
                     lr=2e-5)

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# Number of training epochs (authors recommend between 2 and 4)
epochs = 20

# trange is a tqdm wrapper around the normal python range
for _ in trange(epochs, desc="Epoch"):

    # Training

    # Set our model to training mode (as opposed to evaluation mode)
    model.train()

    # Tracking variables
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    # Train the data for one epoch
    for step, batch in enumerate(train_dataloader):
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Clear out the gradients (by default they accumulate)
        optimizer.zero_grad()
        # Forward pass
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs[0]
        logits = outputs[1]
        # Backward pass
        loss.sum().backward()
        # Update parameters and take a step using the computed gradient
        optimizer.step()

        # Update tracking variables
        tr_loss += loss.mean().item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1

    print("Train loss: {}".format(tr_loss / nb_tr_steps))

# TEST TIME!

prediction_inputs = torch.tensor(input_ids_test)
prediction_masks = torch.tensor(attention_masks_test)
prediction_labels = torch.tensor(labels_test)

prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

# Put model in evaluation mode
model.eval()

# Tracking variables
predictions, true_labels = [], []

# Predict
eval_accuracy, nb_eval_steps = 0, 0
for batch in prediction_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch
    # Telling the model not to compute or store gradients, saving memory and speeding up prediction
    with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        logits = outputs[0]

    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    tmp_eval_accuracy = flat_accuracy(logits, label_ids)

    eval_accuracy += tmp_eval_accuracy
    nb_eval_steps += 1
    round_predictions = [a for a in np.argmax(logits, axis=1).flatten()]
    predictions += round_predictions
    true_labels += [a for a in label_ids.flatten()]

    for i, label in enumerate(round_predictions):
        index = input_ids_map[tuple(b_input_ids[i].detach().cpu().numpy())]
        test_set[index]['answer_binary_embedding_similarity'] = int(label)

print("Test Accuracy: {}".format(eval_accuracy / nb_eval_steps))
print("F1 pos: {}".format(f1_score(true_labels, predictions, pos_label=1)))
print("F1 neg: {}".format(f1_score(true_labels, predictions, pos_label=0)))
print("F1 Macro: {}".format(f1_score(true_labels, predictions, average='macro')))

# with open('../data/ttt/q{}_test.json'.format(file_number), 'w') as f:
#     f.write(json.dumps(test_set))
#
# print('Predicted labels for test portion saved.')
