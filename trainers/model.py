import abc


class Model(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    async def train(self, csv_file_path: str) -> str:
        pass

    @abc.abstractmethod
    def predict(self, model_file_name: str, previous_data: list) -> list:
        pass
