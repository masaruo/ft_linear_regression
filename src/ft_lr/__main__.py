from __future__ import annotations
import argparse
import numpy as np
from src.ft_nn.network import LinearNetwork
from src.ft_nn.optimizers import SGD
from tqdm import trange
import matplotlib.pyplot as plt
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_params(params_path: str) -> tuple[float, float]:
    try:
        with open(params_path, "r") as f:
            params = json.load(f)
            return float(params["theta0"]), float(params["theta1"])
    except Exception:
        return 0.0, 0.0


def save_params(params_path: str, theta0: float = 0, theta1: float = 0) -> None:
    payload = {
        "theta0": theta0,
        "theta1": theta1,
    }
    with open(params_path, "w") as f:
        json.dump(payload, f)


def load_dataset(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    raw = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    if raw.ndim != 2 or raw.shape[1] < 2:
        raise ValueError(f"invalid csv format: {csv_path}")
    x = raw[:, 0].astype(np.float64)
    t = raw[:, 1].astype(np.float64)
    return x, t


def train(
    data_path: str,
    epochs: int,
    lr: float,
) -> None:
    try:
        x, t = load_dataset(data_path)
    except Exception as e:
        logger.info(f"data file not found: {e}")
        return

    network = LinearNetwork()
    optimizer = SGD(lr=lr)

    x_mean = float(np.mean(x))
    x_std = float(np.std(x))
    if x_std == 0.0:
        raise ValueError("x has zero variance")
    x_scaled = (x - x_mean) / x_std  # １標準偏差あたりの変化に変更

    linear_layer = network.layers[0]
    params = [linear_layer.theta0, linear_layer.theta1]

    for _ in trange(epochs):
        for param in params:
            param.grad = 0.0
        network.loss(x_scaled, t)
        network.gradient(x_scaled, t)
        optimizer.update(params)

    theta0, theta1 = to_original_params(network, x_mean, x_std)
    save_params("params.json", theta0, theta1)
    save_plot(x, t, theta0, theta1)


def to_original_params(
    network: LinearNetwork,
    x_mean: float,
    x_std: float,
) -> tuple[float, float]:
    linear_layer = network.layers[0]
    theta1_scaled = linear_layer.theta1.data
    theta0_scaled = linear_layer.theta0.data
    """
    y = t0' + t1' * x'
    x' = (x - mean) / sd
    
    y = t0 + t1 * x'
    y = t0 + (t1 * x)/sd - (t1 * mean)/sd
    y = (t0 - (t1 * mean)/sd) + t1/sd * x
    
    """
    theta1 = theta1_scaled / x_std
    theta0 = theta0_scaled - theta1_scaled * (x_mean / x_std)
    return theta0, theta1


def predict_from_params(x: np.ndarray, theta0: float, theta1: float) -> np.ndarray:
    return theta0 + theta1 * x


def save_plot(
    x: np.ndarray,
    t: np.ndarray,
    theta0: float,
    theta1: float,
) -> None:
    x_line = np.linspace(float(np.min(x)), float(np.max(x)), 200)
    y_line = predict_from_params(x_line, theta0, theta1)

    plt.figure(figsize=(8, 5))
    plt.scatter(x, t, label="data", alpha=0.75)
    plt.plot(x_line, y_line, color="red", label="fit")
    plt.xlabel("km")
    plt.ylabel("price")
    plt.title("Linear Regression Fit")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot.png")
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train linear regression using ft_nn")
    parser.add_argument(
        "--km",
        type=float,
        default=None,
        help="Mileage to forecast price after training",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.km is not None:
        theta0, theta1 = load_params("params.json")
        logger.info(
            f"Mileage:[{args.km}], theta0[{theta0}], theta1[{theta1}] and est price at [{theta0 + theta1 * args.km}]"
        )
        return

    train("data.csv", 50000, 0.01)


if __name__ == "__main__":
    main()
