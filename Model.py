from pandas import DataFrame
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import (TensorDataset, DataLoader, RandomSampler,
                              SequentialSampler)


class Model:

    def __init__(self, token_data: DataFrame, class_data: DataFrame):
        self.token_data = token_data
        self.class_data = class_data
        self.x_test, self.x_train, self.y_test, self.y_train = self.train_test_split()

    def train_test_split(self, test_size: float = 0.3, random_seed: int = 420):
        return train_test_split(
            self.token_data, self.class_data, test_size=test_size, random_state=random_seed)

    def min_max_pair(self):
        return [[self.token_data.loc[:, col].min(), self.token_data.loc[:, col].max()] for col in self.token_data.columns]

    # def data_loader(self, train_inputs, val_inputs, train_labels, val_labels, batch_size=50):
    #     # Convert data type to torch.Tensor
    #     train_inputs, val_inputs, train_labels, val_labels =\
    #         tuple(torch.tensor(data) for data in
    #               [train_inputs, val_inputs, train_labels, val_labels])

    #     # Create DataLoader for training data
    #     train_data = TensorDataset(train_inputs, train_labels)
    #     train_sampler = RandomSampler(train_data)
    #     train_dataloader = DataLoader(
    #         train_data, sampler=train_sampler, batch_size=self.BATCH_SIZE)

    #     # Create DataLoader for validation data
    #     val_data = TensorDataset(val_inputs, val_labels)
    #     val_sampler = SequentialSampler(val_data)
    #     val_dataloader = DataLoader(
    #         val_data, sampler=val_sampler, batch_size=self.BATCH_SIZE)

    #     return train_dataloader, val_dataloader
