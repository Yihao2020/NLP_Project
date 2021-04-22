# import and install required packages

import pandas as pd
import numpy as np
import tensorflow as tf
import torch
from torch.nn import BCEWithLogitsLoss, BCELoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import pickle
from transformers import *
from tqdm import tqdm, trange
from ast import literal_eval
from transformers import BertModel

# two-dense layers on top of BERT
class CustomBERTModel(torch.nn.Module):
    def __init__(self):
          super(CustomBERTModel, self).__init__()
          self.bert = BertModel.from_pretrained("bert-base-uncased", return_dict=False)
          ### New layers:
          self.linear1 = torch.nn.Linear(768, 256)
          self.linear2 = torch.nn.Linear(256, 6) ## 6 is the number of classes in this example

    def forward(self, ids, mask):
          sequence_output, pooled_output = self.bert(ids, attention_mask=mask)

          # sequence_output has the following shape: (batch_size, sequence_length, 768)
          linear1_output = self.linear1(sequence_output[:,0,:].view(-1,768)) ## extract the 1st token's embeddings
          linear2_output = self.linear2(linear1_output)

          return linear2_output

# single dense layer on top of BERT
'''
class CustomBERTModel(torch.nn.Module):
    def __init__(self):
          super(CustomBERTModel, self).__init__()
          self.bert = BertModel.from_pretrained("bert-base-uncased")
          # New Layers
          self.linear1 = torch.nn.Linear(768, 6) ## 6 is the number of classes

    def forward(self, ids, mask):
          sequence_output, pooled_output = self.bert(ids, attention_mask=mask)

          # next, we apply the linear layer. The linear layer (which applies a linear transformation)
          # takes as input the hidden states of all tokens (so seq_len times a vector of size 768, each corresponding to
          # a single token in the input sequence) and outputs 6 numbers (scores, or logits) for every token
          # so the logits are of shape (batch_size, sequence_length, 6)
          logits = self.linear1(sequence_output)

          return logits
'''

def main():
    # reading the file into a dataframe

    df = pd.read_csv('../data/train.csv') # train.csv should be in the same directory as this pyhton file

    '''
    # Exploring Data
    print('Data Frame head')
    print(df.head())

    # Exploring data
    print('Unique comments: ', df.comment_text.nunique() == df.shape[0]) # Checking if there are any duplicate comments
    print('Null values: ', df.isnull().values.any()) # Checking if there are any null comments

    print('average sentence length: ', df.comment_text.str.split().str.len().mean())
    print('stdev sentence length: ', df.comment_text.str.split().str.len().std())

    cols = df.columns
    label_cols = list(cols[2:])
    num_labels = len(label_cols)
    print('Label columns: ', label_cols)
    '''


    # Creating one-hot encodings for each comment. These one-hot encodings will be the target variables for the dataset

    df = df.sample(frac=1).reset_index(drop=True) #shuffle rows
    df['one_hot_labels'] = list(df[label_cols].values)

    #print('Data Frame head after creaeting one-hot encodings')
    #print(df.head())

    labels = list(df.one_hot_labels.values) # Creating a list of the label one-hot encodings
    comments = list(df.comment_text.values) # Creating a list of all comments

    max_length = 128 # keeping maximum length of tokens as 128
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True) # pre-trained BERT tokenizer
    encodings = tokenizer.batch_encode_plus(comments,max_length=max_length,pad_to_max_length=True) # tokenizer's encoding method

    #print('tokenizer outputs: ', encodings.keys())

    input_ids = encodings['input_ids'] # tokenized and encoded sentences
    token_type_ids = encodings['token_type_ids'] # token type ids
    attention_masks = encodings['attention_mask'] # attention masks

    # Identifying indices of 'one_hot_labels' entries that only occur once - this will allow us to stratify split our training data later
    label_counts = df.one_hot_labels.astype(str).value_counts()
    one_freq = label_counts[label_counts==1].keys()
    one_freq_idxs = sorted(list(df[df.one_hot_labels.astype(str).isin(one_freq)].index), reverse=True)
    #print('df label indices with only one instance: ', one_freq_idxs)

    # Gathering single instance inputs to force into the training set after stratified split
    one_freq_input_ids = [input_ids.pop(i) for i in one_freq_idxs]
    one_freq_token_types = [token_type_ids.pop(i) for i in one_freq_idxs]
    one_freq_attention_masks = [attention_masks.pop(i) for i in one_freq_idxs]
    one_freq_labels = [labels.pop(i) for i in one_freq_idxs]

    # Use train_test_split to split our data into train and validation sets

    temp_train_inputs, test_inputs, temp_train_labels, test_labels, temp_train_token_types, test_token_types, temp_train_masks, test_masks = train_test_split(input_ids, labels, token_type_ids,attention_masks,
                                                                random_state=2021, test_size=0.20, stratify = labels)

    train_inputs, validation_inputs, train_labels, validation_labels, train_token_types, validation_token_types, train_masks, validation_masks = train_test_split(temp_train_inputs, temp_train_labels, temp_train_token_types,temp_train_masks,
                                                                random_state=2021, test_size=0.10, stratify = temp_train_labels)

    # Add one frequency data to train data
    train_inputs.extend(one_freq_input_ids)
    train_labels.extend(one_freq_labels)
    train_masks.extend(one_freq_attention_masks)
    train_token_types.extend(one_freq_token_types)

    # Convert all of our data into torch tensors, the required datatype for our model
    train_inputs = torch.tensor(train_inputs)
    train_labels = torch.tensor(train_labels)
    train_masks = torch.tensor(train_masks)
    train_token_types = torch.tensor(train_token_types)

    validation_inputs = torch.tensor(validation_inputs)
    validation_labels = torch.tensor(validation_labels)
    validation_masks = torch.tensor(validation_masks)
    validation_token_types = torch.tensor(validation_token_types)

    test_inputs = torch.tensor(test_inputs)
    test_labels = torch.tensor(test_labels)
    test_masks = torch.tensor(test_masks)
    test_token_types = torch.tensor(test_token_types)

    #The authors recommend a batch size of 32, 48, or 128. We will use 32 here to avoid memory issues.
    batch_size = 32

    # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop,
    # with an iterator the entire dataset does not need to be loaded into memory

    train_data = TensorDataset(train_inputs, train_masks, train_labels, train_token_types)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels, validation_token_types)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    test_data = TensorDataset(test_inputs, test_masks, test_labels, test_token_types)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    # saving
    torch.save(validation_dataloader,'validation_data_loader')
    torch.save(train_dataloader,'train_data_loader')
    torch.save(test_dataloader,'test_data_loader')

    # Load model, the pretrained model will include a single linear classification layer on top for classification.
    #model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

    # Load model, this Custom Model will include one or two dense layers and a classification layer on top of pretrained BERT model.
    model = CustomBERTModel()

    model.cuda()

    optimizer = AdamW(model.parameters(),lr=2e-5)  # Default optimization

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print(torch.cuda.get_device_name(0))

    # Store our loss and accuracy for plotting
    train_loss_set = []

    # Number of training epochs (authors recommend between 2 and 4)
    epochs = 3

    # trange is a tqdm wrapper around the normal python range
    for _ in trange(epochs, desc="Epoch"):

      # Training

      # Set our model to training mode (as opposed to evaluation mode)
      model.train()

      # Tracking variables
      tr_loss = 0 #running loss
      nb_tr_examples, nb_tr_steps = 0, 0

      # Train the data for one epoch
      for step, batch in enumerate(train_dataloader):
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels, b_token_types = batch
        # Clear out the gradients (by default they accumulate)
        optimizer.zero_grad()

        # # Forward pass for multiclass classification
        # outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        # loss = outputs[0]
        # logits = outputs[1]

        # Forward pass for multilabel classification
        #outputs = model(b_input_ids, token_type_ids=None, attention_mask = b_input_mask)

        outputs = model(b_input_ids, b_input_mask)
        #logits = outputs[0]
        logits = outputs

        loss_func = BCEWithLogitsLoss()
        loss = loss_func(logits.view(-1,num_labels),b_labels.type_as(logits).view(-1,num_labels)) #convert labels to float for calculation
        # loss_func = BCELoss()
        # loss = loss_func(torch.sigmoid(logits.view(-1,num_labels)),b_labels.type_as(logits).view(-1,num_labels)) #convert labels to float for calculation
        train_loss_set.append(loss.item())

        # Backward pass
        loss.backward()
        # Update parameters and take a step using the computed gradient
        optimizer.step()
        # scheduler.step()
        # Update tracking variables
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1

      print("Train loss: {}".format(tr_loss/nb_tr_steps))
    ###############################################################################

      # Validation

      # Put model in evaluation mode to evaluate loss on the validation set
      model.eval()

      # Variables to gather full output
      logit_preds,true_labels,pred_labels,tokenized_texts = [],[],[],[]

      # Predict
      for i, batch in enumerate(validation_dataloader):
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels, b_token_types = batch
        with torch.no_grad():
          # Forward pass
          #outs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
          outputs = model(b_input_ids, b_input_mask)
          #b_logit_pred = outs[0]
          b_logit_pred = outputs
          pred_label = torch.sigmoid(b_logit_pred)

          b_logit_pred = b_logit_pred.detach().cpu().numpy()
          pred_label = pred_label.to('cpu').numpy()
          b_labels = b_labels.to('cpu').numpy()

        tokenized_texts.append(b_input_ids)
        logit_preds.append(b_logit_pred)
        true_labels.append(b_labels)
        pred_labels.append(pred_label)

      # Flatten outputs
      pred_labels = [item for sublist in pred_labels for item in sublist]
      true_labels = [item for sublist in true_labels for item in sublist]

      # Calculate Accuracy
      threshold = 0.50
      pred_bools = [pl>threshold for pl in pred_labels]
      true_bools = [tl==1 for tl in true_labels]
      val_f1_accuracy = f1_score(true_bools,pred_bools,average='micro')*100
      val_flat_accuracy = accuracy_score(true_bools, pred_bools)*100

      print('F1 Validation Accuracy: ', val_f1_accuracy)
      print('Flat Validation Accuracy: ', val_flat_accuracy)

    torch.save(model.state_dict(), 'bert_model_two_dense_layer')
    #torch.save(model.state_dict(), 'bert_model_one_dense_layer')

    model.load_state_dict(torch.load('bert_model_two_dense_layer'))

    # Test

    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()
    test_label_cols = list(df.columns[2:])

    #track variables
    logit_preds,true_labels,pred_labels = [],[],[]

    # Predict
    for i, batch in enumerate(test_dataloader):
      batch = tuple(t.to(device) for t in batch)
      # Unpack the inputs from our dataloader
      b_input_ids, b_input_mask, b_labels, b_token_types = batch
      with torch.no_grad():
        # Forward pass
        outs = model(b_input_ids, b_input_mask)
        b_logit_pred = outs[0]
        #b_logit_pred = outs
        pred_label = torch.sigmoid(b_logit_pred)

        b_logit_pred = b_logit_pred.detach().cpu().numpy()
        pred_label = pred_label.to('cpu').numpy()
        b_labels = b_labels.to('cpu').numpy()

      logit_preds.append(b_logit_pred)
      true_labels.append(b_labels)
      pred_labels.append(pred_label)

    # Flatten outputs
    pred_labels = [item for sublist in pred_labels for item in sublist]
    true_labels = [item for sublist in true_labels for item in sublist]
    # Converting flattened binary values to boolean values
    true_bools = [tl==1 for tl in true_labels]

    pred_bools = [pl>0.50 for pl in pred_labels] #boolean output after thresholding

    # Print and save classification report
    print('Test F1 Accuracy: ', f1_score(true_bools, pred_bools,average='micro'))
    print('Test Flat Accuracy: ', accuracy_score(true_bools, pred_bools))
    print('Test Recall: ', recall_score(true_bools, pred_bools,average='micro'))
    print('Test Precision: ', precision_score(true_bools, pred_bools,average='micro'))

    clf_report = classification_report(true_bools,pred_bools,target_names=test_label_cols[:6])

    true_int = np.array(true_bools)
    pred_int = np.array(pred_bools)
    confusion_mtrx = multilabel_confusion_matrix(true_int,pred_int)

    print(clf_report) # Classification Report
    print(confusion_mtrx) # Confusion Matrix

if __name__ == "__main__":
    main()
