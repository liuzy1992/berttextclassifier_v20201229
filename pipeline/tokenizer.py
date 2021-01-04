#!/usr/bin/env python3

from transformers import  BertTokenizer
import torch
from torch.utils.data import TensorDataset

def tokenizer(df, max_seq_length, pretrained_model):
    tokenizer = BertTokenizer.from_pretrained(pretrained_model, do_lower_case=True)
    
    input_ids = []
    attention_masks = []
    
    for text in df['titlecontent']:
        encoded_dict = tokenizer.encode_plus(
                            text,
                            add_special_tokens = True,
                            max_length = max_seq_length,
                            pad_to_max_length = True,
                            return_attention_mask = True,
                            return_tensors = 'pt',
                            truncation=True,
                        )
        
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(df['label'])
    
    dataset = TensorDataset(input_ids, attention_masks, labels)
    return dataset
