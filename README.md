# ft_liner_regression

Simple linear regression training/forecast project using `ft_nn`.

## Setup

```bash
make setup
```

## Common commands

```bash
make train
make predict KM=100000
```

## Output notes (詳細)

### モデル係数

- `theta0`（切片）
	- 回帰式 `price = theta0 + theta1 * km` における定数項です。
	- `km=0` のときの理論上の予測価格を表します。
- `theta1`（傾き）
	- `km` が 1 増えるごとに価格がどれだけ変化するかを表します。
	- 例: `theta1 = -0.02` なら、走行距離が 1km 増えるごとに価格が約 0.02 下がる解釈です。
	- 中古車データでは通常マイナスになるのが自然です。

### 予測値

- `forecast_price`
	- 指定した `--km` に対する予測価格です。
	- 計算は `theta0 + theta1 * km` で行われます。

<!-- ### 誤差指標（train/test）

- `train_mae`, `test_mae`
	- 平均絶対誤差（MAE）で、式は `mean(|y - y_hat|)` です。
	- 単位は価格そのもの（円）で、直感的に「平均でどれくらいズレるか」を示します。
	- 小さいほど良いです。

- `train_rmse`, `test_rmse`
	- 二乗平均平方根誤差（RMSE）で、式は `sqrt(mean((y - y_hat)^2))` です。
	- 大きな誤差をより強く罰するため、外れ値の影響を受けやすいです。
	- 小さいほど良いです。

- `train_r2`, `test_r2`
	- 決定係数（R²）で、式は `1 - SS_res / SS_tot`
	平均値で予測するより、回帰での予測がどれだけ誤差を減らせたか
	- `SS_res`（残差平方和）は `Σ (y_i - y_pred_i)^2` です。
		- 実測値と予測値のズレを 2 乗して合計したものです。
	- `SS_tot`（全変動平方和）は `Σ (y_i - y_mean)^2` です。
		- 実測値と平均値のズレを 2 乗して合計したものです。
	- 1.0 に近いほど説明力が高く、0.0 は「平均予測と同程度」、マイナスは「平均予測より悪い」を意味します。
	- 大きいほど良いです。 -->

<!-- ### baseline（基準モデル）

- `baseline_test_*`
	- 学習モデルではなく、「訓練データの平均価格を常に予測するだけ」の単純モデルです。
	- モデルの強さは、`test_mae` / `test_rmse` が `baseline_test_mae` / `baseline_test_rmse` より小さいかで確認できます。
	- さらに `test_r2` が baseline より高い（できれば正の値）なら、より有効なモデルと判断しやすいです。

### 強さを判断するときの見方

- 過学習チェック:
	- `train_*` が良いのに `test_*` が悪い場合は過学習の可能性があります。
- 実用性チェック:
	- `test_mae` が業務上許容できる誤差幅に収まっているかを確認します。
- 安定性チェック:
	- `SEED` を変えて `make eval` を複数回実行し、指標のブレが小さいかを見ると信頼しやすいです。 -->
