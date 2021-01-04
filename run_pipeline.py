#!/usr/bin/env python3

import sys
from pipeline import *

def main(infile, 
         pretrained_model, 
         max_length, 
         batch_size, 
         num_epochs, 
         learning_rate,
         model_outdir):
    df_train, df_valid, df_test = preprocessing(infile)
    train_dataset = tokenizer(df_train, max_length, pretrained_model)
    valid_dataset = tokenizer(df_valid, max_length, pretrained_model)
    test_dataset = tokenizer(df_test, max_length, pretrained_model)
    train_dataloader, valid_dataloader, test_dataloader = dataloader(train_dataset, 
                                                                     valid_dataset, 
                                                                     test_dataset, 
                                                                     batch_size)
    training(train_dataloader, 
             valid_dataloader, 
             pretrained_model, 
             num_epochs,
             learning_rate,
             model_outdir)
    evaluation(test_dataloader, pretrained_model, model_outdir)

main(infile=sys.argv[1], 
     pretrained_model=sys.argv[2], 
     max_length=int(sys.argv[3]), 
     batch_size=int(sys.argv[4]), 
     num_epochs=int(sys.argv[5]), 
     learning_rate=float(sys.argv[6]),
     model_outdir=sys.argv[7])

