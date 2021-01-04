#!/usr/bin/env python3

import torch
from . import device


def save_model(save_path, model, valid_loss):

    if save_path == None:
        return

    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}

    torch.save(state_dict, save_path)
    print('Model saved to ==> ' + save_path)

def load_model(load_path, model):

    if load_path==None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print('Model loaded from <== ' + load_path)

    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']

