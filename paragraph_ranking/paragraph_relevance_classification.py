import json
import numpy as np
import random
import sys
import torch
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import trange
from transformers import AdamW
from transformers import XLNetTokenizer, XLNetForSequenceClassification


def train_and_label(train_set, test_set, device, n_gpu, epochs):
    # Create sentence and label lists
    sentences_train = [item['paragraph'] + " [SEP] [CLS]" for item in train_set]
    labels_train = [item['label'] for item in train_set]

    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=False)

    tokenized_texts_train = [tokenizer.tokenize(sent) for sent in sentences_train]
    print("Tokenize the first sentence:")
    print(tokenized_texts_train[0])

    # Set the maximum sequence length. The longest sequence in our training set is 47, but we'll leave room on the end anyway.
    MAX_LEN = 280
    average_len = 0
    reduced_inputs = 0
    for tokens in tokenized_texts_train:
        average_len += len(tokens)
        if len(tokens) > MAX_LEN:
            reduced_inputs += 1

    average_len /= len(tokenized_texts_train)

    print("reduced input is: {}".format(reduced_inputs))
    print("average_len is: {}".format(average_len))

    # Use the XLNet tokenizer to convert the tokens to their index numbers in the XLNet vocabulary
    input_ids_train = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts_train]

    # Pad our input tokens
    input_ids_train = pad_sequences(input_ids_train, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # Create attention masks
    attention_masks_train = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids_train:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks_train.append(seq_mask)

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

    model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=2)
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

        # We now tag the test data with what we have learned so far
        # Create sentence lists from paragraphs
        test_paragraphs = []
        paragraphs_map = {}
        for i, article in enumerate(test_set):
            article['paragraph_relevance_learned_labels'] = [0] * len(article['paragraphs'])
            test_paragraphs.extend(article['paragraphs'])
            for j, paragraph in enumerate(article['paragraphs']):
                paragraphs_map[paragraph] = [i, j]

        sentences_test = [paragraph + " [SEP] [CLS]" for paragraph in test_paragraphs]

        tokenized_texts_test = [tokenizer.tokenize(sent) for sent in sentences_test]

        # Use the XLNet tokenizer to convert the tokens to their index numbers in the XLNet vocabulary
        input_ids_test = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts_test]

        # Pad our input tokens
        input_ids_test = pad_sequences(input_ids_test, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
        input_ids_map = {}
        for i, point in enumerate(input_ids_test):
            input_ids_map[tuple(point)] = i

        # Create attention masks
        attention_masks_test = []

        # Create a mask of 1s for each token followed by 0s for padding
        for seq in input_ids_test:
            seq_mask = [float(i > 0) for i in seq]
            attention_masks_test.append(seq_mask)

        prediction_inputs = torch.tensor(input_ids_test)
        prediction_masks = torch.tensor(attention_masks_test)

        prediction_data = TensorDataset(prediction_inputs, prediction_masks)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

        # Predict
        for batch in prediction_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                logits = outputs[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            predictions = [a for a in np.argmax(logits, axis=1).flatten()]
            for i, label in enumerate(predictions):
                p = test_paragraphs[input_ids_map[tuple(b_input_ids[i].detach().cpu().numpy())]]
                test_set[paragraphs_map[p][0]]['paragraph_relevance_learned_labels'][paragraphs_map[p][1]] = label


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print("GPU name: " + torch.cuda.get_device_name(0))

print("Loading data...")

with open('../data/paragraph_relevance_classification_input/q{}_train.json'.format(sys.argv[1])) as train_file:
    train_set = json.load(train_file)

with open('../data/paragraph_relevance_classification_input/q{}_test.json'.format(sys.argv[1])) as test_file:
    test_set = json.load(test_file)

print("Data loading completed.")

initial_train_set_one = []
initial_train_set_zero = []
rest_of_training_data = []

for article in train_set:
    if 'paragraph_relevance_extracted_labels' not in article:
        rest_of_training_data.append(article)
        continue
    for i, paragraph in enumerate(article['paragraphs']):
        if article['paragraph_relevance_extracted_labels'][i]:
            initial_train_set_one.append({'paragraph': paragraph, 'label': 1})
        elif len(initial_train_set_zero) < len(initial_train_set_one):
            initial_train_set_zero.append({'paragraph': paragraph, 'label': 0})

print('Initial train set stats: 0s count: {}, 1s count: {}'.format(len(initial_train_set_zero),
                                                                   len(initial_train_set_one)))

initial_train_set = initial_train_set_one + initial_train_set_zero
random.Random(2017).shuffle(initial_train_set)

# train on part of train data, label all train data with what you learned.
train_and_label(initial_train_set, rest_of_training_data, device, n_gpu, 3)

# preparing newly labeled data for retraining
second_train_set_one = []
second_train_set_zero = []

for article in train_set:
    if 'paragraph_relevance_learned_labels' not in article:
        continue
    for i, paragraph in article['paragraphs']:
        if article['paragraph_relevance_learned_labels'][i]:
            second_train_set_one.append({'paragraph': paragraph, 'label': 1})
        elif len(second_train_set_zero) < len(second_train_set_one):
            second_train_set_zero.append({'paragraph': paragraph, 'label': 0})

second_train_set = second_train_set_one + second_train_set_zero
random.Random(2017).shuffle(second_train_set)

# train on training data, label test data
train_and_label(second_train_set, test_set, device, n_gpu, 15)

with open('../data/paragraph_relevance_classification_input/q{}_train.json'.format(sys.argv[1]), 'w') as f:
    f.write(json.dumps(train_set))

with open('../data/paragraph_relevance_classification_input/q{}_test.json'.format(sys.argv[1]), 'w') as f:
    f.write(json.dumps(test_set))