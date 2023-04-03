from pandas import DataFrame
from sklearn.model_selection import train_test_split


class model:
    def __init__(self, data: DataFrame):
        self.data = data
        self.x_test = None
        self.x_train = None
        self.y_test = None
        self.y_train = None

    def get_data_input(self):
        return self.data.iloc[:, :-1]

    def get_data_output(self):
        return self.data.iloc[:, -1:]

    def train_test_split(self, test_size: float = 0.3, random_seed: int = 0):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            get_data_input(), get_data_output(), test_size=test_size, random_state=random_seed)
