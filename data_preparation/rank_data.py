import torch, json
import torch.nn as nn
from torch.utils import data

from paragraph_ranking import get_paragraph_similarities

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
embeddings_size = 768
input_size = (embeddings_size + 1) * 50
hidden_size = 500
num_classes = 50
num_epochs = 5
batch_size = 1
learning_rate = 0.001

with open('../data/ranking/q1_train.json') as q1_train_file:
    q1_train = json.load(q1_train_file)

with open('../data/ranking/q1_test.json') as q1_test_file:
    q1_test = json.load(q1_test_file)

with open('../data/ranking/q1_dev.json') as q1_dev_file:
    q1_test += json.load(q1_dev_file)


class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, articles):
        'Initialization'
        self.articles = articles

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.articles)

    def __getitem__(self, index):
        'Generates one sample of data'
        a = self.articles[index]
        exp_similarities, q_similarities, paragraphs_embeddings, paragraphs = get_paragraph_similarities(a['article'],
                                                                                                         a['question'],
                                                                                                         a[
                                                                                                             'explanation'])
        X = torch.zeros(input_size)
        for i, paragraph_embeddings in enumerate(paragraphs_embeddings):
            X[i * embeddings_size: (i + 1) * embeddings_size] = paragraph_embeddings

        # paddings
        padding_length = embeddings_size * num_classes - embeddings_size * len(paragraphs_embeddings)
        if padding_length > 0:
            X[embeddings_size * len(paragraphs_embeddings): embeddings_size * num_classes] = torch.tensor([-1] * padding_length)

        X[embeddings_size * num_classes:] = torch.tensor(q_similarities + [-1] * (num_classes - len(paragraphs_embeddings)))

        y = torch.tensor(exp_similarities + [-1] * (num_classes - len(paragraphs_embeddings)), dtype=torch.long)

        return X, y


q1_train_transformed = Dataset(q1_train)
q1_test_transformed = Dataset(q1_test)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=q1_train_transformed,
                                           batch_size=batch_size,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=q1_test_transformed,
                                          batch_size=batch_size,
                                          shuffle=False)


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (embeddings, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        embeddings = embeddings.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(embeddings)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for embeddings, labels in test_loader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        outputs = model(embeddings)
        # _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (outputs == labels).sum().item()

    print('Accuracy of the network: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
