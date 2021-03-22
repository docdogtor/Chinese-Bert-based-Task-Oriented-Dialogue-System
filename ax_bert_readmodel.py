import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange
import tensorflow as tf
import io
import os

#import matplotlib.pyplot as plt
#% matplotlib inline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

# #differnent on google drive when using google colab
# print(os.listdir("atis"))
# import pickle
# #differnent on google drive when using google colab
# DATA_DIR="atis"

output_dir = './model_save/'

# Example for a Bert model
model = BertForSequenceClassification.from_pretrained(output_dir, num_labels=26)
tokenizer = BertTokenizer.from_pretrained(output_dir, do_lower_case=True)  # Add specific options if needed
model.to(device)

load_test = open('test_data.txt','r')
lines = load_test.readlines()
query_data_test = []
intent_data_test = []
intent_data_label_test = []
for line in lines:
    line = line.split("\t")
    intent_data_test.append(line[0])
    if line[0] == '机票预订':
        intent_data_label_test.append(1)
    elif line[0] == '餐厅预订':
        intent_data_label_test.append(2)
    else:
        intent_data_label_test.append(3)
    query_data_test.append(line[1])
load_test.close()

sentences = ["[CLS] " + query + " [SEP]" for query in query_data_test]
labels = intent_data_label_test
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
MAX_LEN = 128
input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
attention_masks = []
for seq in input_ids:
    seq_mask = [float(i > 0) for i in seq]
    attention_masks.append(seq_mask)

prediction_inputs = torch.tensor(input_ids)
prediction_masks = torch.tensor(attention_masks)
prediction_labels = torch.tensor(labels)
batch_size = 48
prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

model.eval()
predictions, true_labels = [], []
for batch in prediction_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch
    with torch.no_grad():
        logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    predictions.append(logits)
    true_labels.append(label_ids)

#print(intent_data_label_test)
#'''correct labels'''

from sklearn.metrics import matthews_corrcoef

matthews_set = []
for i in range(len(true_labels)):
    matthews = matthews_corrcoef(true_labels[i],
                                    np.argmax(predictions[i], axis=1).flatten())
    matthews_set.append(matthews)

flat_predictions = [item for sublist in predictions for item in sublist]
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
flat_true_labels = [item for sublist in true_labels for item in sublist]

# print(flat_predictions)
# '''[0, 1] for different sentences'''
if flat_predictions[0] == 1:
    print("机票预订")
if flat_predictions[0] == 2:
    print("餐厅预订")
if flat_predictions[0] == 3:
    print("其他")
# print('Classification accuracy using BERT Fine Tuning: {0:0.2%}'.format(matthews_corrcoef(flat_true_labels, flat_predictions)))
