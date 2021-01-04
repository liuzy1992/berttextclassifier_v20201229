#!/usr/bin/env python3

import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from .preprocessing import preprocessing
from .tokenizer import tokenizer
from .dataloader import dataloader
from .training import training
from .evaluation import evaluation
