import numpy as np
from .layers import SumOfSquaredError, LinearLayer
import logging

logger = logging.getLogger(__name__)


class LinearNetwork:
    def __init__(self):
        self.layers = []
        self.layers.append(LinearLayer())
        self.last_layer = SumOfSquaredError()

    def predict(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def loss(self, x: np.ndarray, t: np.ndarray) -> float:
        y = self.predict(x)
        loss = self.last_layer.forward(y, t)
        return loss

    def gradient(self, x: np.ndarray, t: np.ndarray) -> None:
        self.loss(x, t)
        dout = self.last_layer.backward(1.0)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
