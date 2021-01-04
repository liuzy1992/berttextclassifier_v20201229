#!/usr/bin/env python3

import time
import torch
from transformers import BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from . import device
from .savingandloading import save_model

def training(train_dataloader, 
             valid_dataloader, 
             pretrained_model, 
             num_epochs, 
             lr,
             file_path):
    model = BertForSequenceClassification.from_pretrained(pretrained_model,
                                                          num_labels = 2,
                                                          output_attentions = False,
                                                          output_hidden_states = False)
    optimizer = AdamW(model.parameters(), lr = lr, eps = 1e-8)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=0, 
                                                num_training_steps=total_steps)

    train_loss_list = []
    valid_loss_list = []
    train_r2_list = []
    valid_r2_list = []
    epoch_list = []

    best_valid_loss = float("Inf")
    total_t0 = time.time()
    
    for epoch in range(0, num_epochs):
        t0 = time.time()
        total_train_loss = 0
        total_train_r2 = 0

        model.train()

        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            model.zero_grad()
            
            loss, logits = model(
                b_input_ids,
                token_type_ids = None,
                attention_mask = b_input_mask,
                labels = b_labels
            )
            
            total_train_loss += loss.item()
            # total_train_r2 += r2_score(b_labels.tolist(), logits.detach().tolist())
            total_train_r2 += r2_score(b_labels.tolist(), torch.argmax(logits, 1).tolist())
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_r2 = total_train_r2 / len(train_dataloader)
        
        model.eval()
        
        total_val_loss = 0
        total_val_r2 = 0
        
        for batch in valid_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            with torch.no_grad():
                loss, logits = model(
                    b_input_ids,
                    token_type_ids = None,
                    attention_mask = b_input_mask,
                    labels = b_labels
                )
                
            total_val_loss += loss.item()
            total_val_r2 += r2_score(b_labels.tolist(), torch.argmax(logits, 1).tolist())
            
        avg_val_loss = total_val_loss / len(valid_dataloader)
        avg_val_r2 = total_val_r2 / len(valid_dataloader)

        print("## Epoch {:}/{:} ==> Train Loss: {:.5f}, Train R2: {:.5f}; Valid Loss: {:.5f}, Valid R2: {:.5f}; Elapsed Time: {:.2f} s".format(
                        epoch + 1,
                        num_epochs,
                        avg_train_loss,
                        avg_train_r2,
                        avg_val_loss,
                        avg_val_r2,
                        time.time() - t0
                    ))
        
        train_loss_list.append(avg_train_loss)
        train_r2_list.append(avg_train_r2)
        valid_loss_list.append(avg_val_loss)
        valid_r2_list.append(avg_val_r2)
        epoch_list.append(epoch + 1)

        if best_valid_loss > avg_val_loss:
            best_valid_loss = avg_val_loss
            save_model(file_path + '/' + 'model.pt', model, best_valid_loss)

    print("Training Complete! Total Elapsed Time: {:.2f} s.".format(time.time() - total_t0))

    plt.plot(epoch_list, train_loss_list, label='Train Loss')
    plt.plot(epoch_list, train_r2_list, label='Train R2')
    plt.plot(epoch_list, valid_loss_list, label='Valid Loss')
    plt.plot(epoch_list, valid_r2_list, label='Valid R2')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
