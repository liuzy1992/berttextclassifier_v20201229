#!/usr/bin/env python3

import torch
from transformers import BertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from . import device
from .savingandloading import load_model

def evaluation(test_dataloader, pretrained_model, file_path):
    y_pred = []
    y_true = []
    
    model = BertForSequenceClassification.from_pretrained(pretrained_model,
                                                          num_labels = 2,
                                                          output_attentions = False,
                                                          output_hidden_states = False)
    load_model(file_path + '/model.pt', model)

    model.eval()
    
    for batch in test_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
            
        with torch.no_grad():
            _, logits = model(
                b_input_ids,
                token_type_ids = None,
                attention_mask = b_input_mask,
                labels = b_labels
            )
        
        y_pred.extend(torch.argmax(logits, 1).tolist())
        y_true.extend(b_labels.tolist())

    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1,0], digits=4))

    cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

    ax.set_title('Confusion Matrix')

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    ax.xaxis.set_ticklabels(['0', '1'])
    ax.yaxis.set_ticklabels(['0', '1'])


