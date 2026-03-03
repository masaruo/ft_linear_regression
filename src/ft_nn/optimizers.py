from .parameter import Parameter


class SGD:
    def __init__(self, lr: float = 0.001):
        self.lr = lr

    def update(self, params: list[Parameter]) -> None:
        for param in params:
            param.data -= self.lr * param.grad
