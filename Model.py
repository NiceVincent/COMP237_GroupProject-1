from pandas import DataFrame
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch


class Model:
    TEST_SIZE = 0.3
    RANDOM_SEED = 0

    def __init__(self, token_data: DataFrame, class_data: DataFrame):
        self.token_data = token_data
        self.class_data = class_data
        self.x_test = None
        self.x_train = None
        self.y_test = None
        self.y_train = None

        # Check here for GPU support https://towardsdatascience.com/installing-pytorch-on-apple-m1-chip-with-gpu-acceleration-3351dc44d67c
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f'There are {torch.cuda.device_count()} GPU(s) available.')
            print('Device name:', torch.cuda.get_device_name(0))

        else:
            print('No GPU available, using the CPU instead.')
            self.device = torch.device("cpu")

    def train_test_split(self, test_size: float = TEST_SIZE, random_seed: int = RANDOM_SEED):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.token_data, self.class_data, test_size=test_size, random_state=random_seed)
        
    def CNN_NLP():
        