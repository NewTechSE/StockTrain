import abc


class Model(metaclass=abc.ABCMeta):
    model_file_name = ""

    @abc.abstractmethod
    async def train(self, csv_file_path: str) -> str:
        pass

    @abc.abstractmethod
    async def predict(self, previous_data: list) -> list:
        pass
