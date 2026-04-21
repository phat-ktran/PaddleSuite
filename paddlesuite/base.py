from abc import ABC, abstractmethod
from typing import Union, List

class BasePredictor(ABC):
    @abstractmethod
    def predict(self, data: Union[str, List[str]]):
        pass

    @abstractmethod
    def predict_iter(self, data: Union[str, List[str]]):
        pass
