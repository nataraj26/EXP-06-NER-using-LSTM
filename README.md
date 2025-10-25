# Named Entity Recognition

## AIM

To develop an LSTM-based model for recognizing the named entities in a given text.

## Problem Statement and Dataset

The goal of this experiment is to design and train a Bidirectional LSTM (BiLSTM) model to identify and classify entities such as persons, organizations, locations, and dates within sentences.
The dataset used for training consists of tagged text sequences, where each word is labeled with its corresponding NER tag (e.g., B-PER, I-LOC, O, etc.).
The data is preprocessed and split into training and testing sets for model evaluation.

## DESIGN STEPS
#### STEP 1:

Import the necessary libraries and load the NER dataset.

#### STEP 2:

Preprocess the dataset — tokenize the text, create vocabulary, and encode the NER tags.

#### STEP 3:

Define the BiLSTMTagger class with embedding, dropout, LSTM, and fully connected layers.

#### STEP 4:

Initialize the model, define the loss function and optimizer for training.

#### STEP 5:

Train the model using the training data and validate using the test set.

#### STEP 6:

Evaluate the model’s performance using loss values and entity recognition accuracy.

#### STEP 7:

Visualize training and validation loss to analyze model performance.

## PROGRAM
### Name: S NATARAJ KUMARAN 
### Register Number: 212223230137
```python
class BiLSTMTagger(nn.Module):

    def __init__(self, vocab_size, tagset_size, embedding_dim=50, hidden_dim=100):
        super(BiLSTMTagger, self).__init__()
        self.embedding=nn.Embedding(vocab_size,embedding_dim)
        self.dropout=nn.Dropout(0.1)
        self.lstm=nn.LSTM(embedding_dim,hidden_dim,batch_first=True,bidirectional=True)
        self.fc=nn.Linear(hidden_dim*2,tagset_size)
    def forward(self,x):
        x=self.embedding(x)
        x=self.dropout(x)
        x,_=self.lstm(x)
        return self.fc(x)


model=BiLSTMTagger(len(word2idx)+1,len(tag2idx)).to(device)
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)


# Training and Evaluation Functions
def train_model(model,train_loader,test_loader,loss_fn,optimixer,epochs=3):
  train_losses,val_losses=[],[]
  for epoch in range(epochs):
    model.train()
    total_loss=0
    for batch in train_loader:
      input_ids=batch["input_ids"].to(device)
      labels=batch["labels"].to(device)
      optimizer.zero_grad()
      outputs=model(input_ids)
      loss=loss_fn(outputs.view(-1,len(tag2idx)),labels.view(-1))
      loss.backward()
      optimizer.step()
      total_loss+=loss.item()
    train_losses.append(total_loss)

    model.eval()
    val_loss=0
    with torch.no_grad():
      for batch in test_loader:
        input_ids=batch["input_ids"].to(device)
        labels=batch["labels"].to(device)
        outputs=model(input_ids)
        loss=loss_fn(outputs.view(-1,len(tag2idx)),labels.view(-1))
        val_loss+=loss.item()
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}: Train Loss = {total_loss:.4f}, Val Loss = {val_loss:.4f}")

  return train_losses,val_losses

```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

<img width="1287" height="601" alt="image" src="https://github.com/user-attachments/assets/8ce7cb5a-fe55-4a90-b08f-27e9e2a3981a" />


### Sample Text Prediction
<img width="1023" height="561" alt="image" src="https://github.com/user-attachments/assets/a3ae5be4-db09-482b-b77d-cf509b9f4bfe" />


## RESULT
The BiLSTM-based NER model was successfully trained.
The training and validation losses decreased over epochs, indicating that the model effectively learned to identify named entities from text sequences.
