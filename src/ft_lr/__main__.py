from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from src.ft_nn.network import LinearNetwork
from src.ft_nn.optimizers import SGD
from tqdm import trange


def load_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    if raw.ndim != 2 or raw.shape[1] < 2:
        raise ValueError(f"invalid csv format: {csv_path}")
    x = raw[:, 0].astype(np.float64)
    t = raw[:, 1].astype(np.float64)
    return x, t


def train_test_split(
    x: np.ndarray,
    t: np.ndarray,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between 0 and 1")
    if len(x) < 2:
        raise ValueError("dataset must contain at least 2 samples")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(x))
    rng.shuffle(indices)

    test_size = max(1, int(len(x) * test_ratio))
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    if len(train_indices) == 0:
        raise ValueError("train split is empty, reduce test_ratio")

    return x[train_indices], t[train_indices], x[test_indices], t[test_indices]


def train(
    x: np.ndarray,
    t: np.ndarray,
    epochs: int,
    lr: float,
) -> tuple[LinearNetwork, float, float, float]:
    network = LinearNetwork()
    optimizer = SGD(lr=lr)

    x_mean = float(np.mean(x))
    x_std = float(np.std(x))
    if x_std == 0.0:
        raise ValueError("x has zero variance")
    x_scaled = (x - x_mean) / x_std #１標準偏差あたりの変化に変更

    linear_layer = network.layers[0]
    params = [linear_layer.theta0, linear_layer.theta1]

    loss = 0.0
    for _ in trange(epochs):
        for param in params:
            param.grad = 0.0
        loss = network.loss(x_scaled, t)
        network.gradient(x_scaled, t)
        optimizer.update(params)

    return network, float(loss), x_mean, x_std


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
    x' = (-mean / sd) + (x / std)
    y = t0 - (t1 * u / std) + (t1 / std) * x
    """
    theta1 = theta1_scaled / x_std
    theta0 = theta0_scaled - theta1_scaled * (x_mean / x_std)
    return theta0, theta1


def predict_from_params(x: np.ndarray, theta0: float, theta1: float) -> np.ndarray:
    return theta0 + theta1 * x


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    error = y_true - y_pred
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0.0 else 0.0
    return mae, rmse, r2


def save_plot(
    x: np.ndarray,
    t: np.ndarray,
    theta0: float,
    theta1: float,
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "matplotlib is required for plotting. Run: uv add matplotlib"
        ) from error

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
    plt.savefig(output_path)
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train linear regression using ft_nn")
    parser.add_argument(
        "--data", type=Path, default=Path("data.csv"), help="Path to csv file"
    )
    parser.add_argument(
        "--epochs", type=int, default=50000, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument(
        "--km",
        type=float,
        default=None,
        help="Mileage to forecast price after training",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate model quality with train/test split",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Test split ratio for --eval",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Save regression plot to image file (e.g. plot.png)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    x, t = load_dataset(args.data)
    if args.eval:
        x_train, t_train, x_test, t_test = train_test_split(
            x, t, test_ratio=args.test_ratio, seed=args.seed
        )
    else:
        x_train, t_train = x, t
        x_test, t_test = x, t

    network, loss, x_mean, x_std = train(
        x_train, t_train, epochs=args.epochs, lr=args.lr
    )
    theta0, theta1 = to_original_params(network, x_mean, x_std)

    print(f"final_loss={loss:.6f}")
    print(f"theta0={theta0:.6f}")
    print(f"theta1={theta1:.6f}")

    if args.eval:
        train_pred = predict_from_params(x_train, theta0, theta1)
        test_pred = predict_from_params(x_test, theta0, theta1)
        train_mae, train_rmse, train_r2 = regression_metrics(t_train, train_pred)
        test_mae, test_rmse, test_r2 = regression_metrics(t_test, test_pred)

        baseline = np.full_like(t_test, np.mean(t_train), dtype=np.float64)
        base_mae, base_rmse, base_r2 = regression_metrics(t_test, baseline)

        print(f"train_mae={train_mae:.6f}")
        print(f"train_rmse={train_rmse:.6f}")
        print(f"train_r2={train_r2:.6f}")
        print(f"test_mae={test_mae:.6f}")
        print(f"test_rmse={test_rmse:.6f}")
        print(f"test_r2={test_r2:.6f}")
        print(f"baseline_test_mae={base_mae:.6f}")
        print(f"baseline_test_rmse={base_rmse:.6f}")
        print(f"baseline_test_r2={base_r2:.6f}")

    if args.km is not None:
        predicted_price = theta0 + theta1 * args.km
        print(f"forecast_km={args.km:.0f}")
        print(f"forecast_price={predicted_price:.6f}")

    if args.plot is not None:
        save_plot(x, t, theta0, theta1, args.plot)
        print(f"plot={args.plot}")


if __name__ == "__main__":
    main()
