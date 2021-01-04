#!/usr/bin/env python3

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

def dataloader(train_dataset, valid_dataset, test_dataset, batch_size):
    train_dataloader = DataLoader(
            train_dataset,
            sampler = RandomSampler(train_dataset),
            batch_size = batch_size
        )

    valid_dataloader = DataLoader(
            valid_dataset,
            sampler = SequentialSampler(valid_dataset),
            batch_size = batch_size
        )

    test_dataloader = DataLoader(
            test_dataset,
            sampler = SequentialSampler(test_dataset),
            batch_size = batch_size
        )

    return train_dataloader, valid_dataloader, test_dataloader
