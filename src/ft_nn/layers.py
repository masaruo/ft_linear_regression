import numpy as np
import logging
from .parameter import Parameter

logger = logging.getLogger(__name__)


class LinearLayer:
    def __init__(self):
        self.theta0 = Parameter(0.0)  # intercept
        self.theta1 = Parameter(0.0)  # slope
        self.x_cache: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        y (car_price) = theta0 + theta1 * x
        """
        self.x_cache = x
        out = self.theta0.data + self.theta1.data * x
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x = self.x_cache
        if x is None:
            raise RuntimeError("backward must be called after forward")
        """theta0
        dy / d_theta0 = 1
        """
        self.theta0.grad += np.sum(dout * 1.0)
        """theta1
        dy / d_theta1 = x
        """
        self.theta1.grad += np.sum(dout * x)

        dx = dout * self.theta1.data
        return dx


class SumOfSquaredError:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, y: np.ndarray, t: np.ndarray) -> float:
        self.y = np.atleast_2d(y)
        self.t = np.atleast_2d(t)

        self.loss = 0.5 * np.sum((self.y - self.t) ** 2)
        return self.loss

    def backward(self, dout: float = 1.0) -> np.ndarray:
        if self.y is None or self.t is None:
            raise RuntimeError("backward must be called after forward")
        """
        J = 0.5(Yi - Ti)^2
        L = SUM(J)

        dL / dY = (dL/dJ) * (dJ/dY)
        dJ / dY = Yi - Ti
        dL / dJ = 1
        """
        dx = dout * (self.y - self.t)
        return dx
