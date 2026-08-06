# Telco Customer Churn Predictor

A Keras MLP that predicts telecom customer churn from 26 account and service features. It reaches
77.9% test accuracy, but it only catches 47% of the customers who actually churn. The second number
is the one that matters.

## Get the data first

**The dataset is not in this repo.** The notebook will fail on the first cell without it.

1. Download **Telco Customer Churn** from Kaggle:
   https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Put `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the repo root, next to `ChurnPredictor.ipynb`.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow jupyter
jupyter notebook ChurnPredictor.ipynb
```

Then run all cells. Training is 100 epochs on 5,625 rows and finishes in well under a minute on CPU.

## Results

All numbers below come from the committed notebook outputs, on a 1,407 row held-out test set
(80/20 split, `random_state=5`).

| Metric | Value |
| --- | --- |
| Test accuracy | 77.9% |
| Test loss (binary crossentropy) | 0.480 |
| Churn recall | 47.3% (193 of 408) |
| Churn precision | 66.8% (193 of 289) |
| Churn F1 | 0.55 |
| No-churn recall | 90.4% |
| No-churn precision | 80.8% |

Confusion matrix, rows are truth and columns are prediction:

|  | Predicted no churn | Predicted churn |
| --- | --- | --- |
| **Actually no churn** | 903 | 96 |
| **Actually churn** | 215 | 193 |

### Why accuracy is the wrong headline

408 of the 1,407 test customers churned, so the classes split 71/29. Always predicting "no churn"
scores 71.0% accuracy while being useless. This model scores 77.9%, so it buys 6.9 points over that
baseline.

What it actually does is miss 215 of 408 churners. For a retention team, those 215 are the entire
point of the exercise, and they never get flagged. The model is conservative at the default 0.5
threshold: it fires rarely and is right two thirds of the time when it does. Lowering the threshold
would trade precision for recall, which is usually the right trade when a retention offer is cheap
and a lost customer is not. That sweep is not in the notebook yet.

### One note on the numbers

Cells near the end of the notebook hand-compute metrics from `862 / 137 / 179 / 229`. Those are from
an earlier training run and no longer match anything else in the notebook. The confusion matrix
plotted directly above them shows `903 / 96 / 215 / 193`, and that one reconciles exactly with the
committed `model.evaluate` output (1,096 correct of 1,407 = 0.77896). Every figure in this README
comes from the plotted matrix.

## Pipeline

1. Load 7,043 rows and 21 columns. Drop `customerID`.
2. `TotalCharges` arrives as text with 11 blank entries. Drop those rows, cast to float. 7,032 rows
   remain.
3. Collapse `No internet service` and `No phone service` to `No`, since they encode the same absence
   as the plain `No` and would otherwise split the feature space three ways.
4. Map the binary Yes/No columns to 1/0. One-hot encode `InternetService`, `Contract`, and
   `PaymentMethod`.
5. Min-max scale `tenure`, `MonthlyCharges`, and `TotalCharges`.
6. Split 80/20. 5,625 train rows, 1,407 test rows, 26 features.

## Model

```python
model = keras.Sequential([
    keras.layers.Dense(26, input_shape=(26,), activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100)
```

No regularisation, no dropout, no early stopping, no validation split. Training accuracy is 0.812 by
epoch 10 and 0.836 at epoch 100, against 0.779 on the test set, so the run is overfitting from very
early on and the last 90 epochs buy 2.4 points of training accuracy and nothing on test.

## Known problems

- **Scaler leakage.** `MinMaxScaler` is fit on the full dataset before the train/test split, so test
  set minima and maxima leak into the training features. The effect is small for min-max on these
  three columns, but the fit belongs after the split.
- **No class weighting.** The 71/29 imbalance is never addressed, which is the direct cause of the
  47% churn recall.
- **No threshold tuning.** Predictions are cut at a hard 0.5.
- **No baseline comparison.** Logistic regression or gradient boosting on the same 26 features would
  likely match this, and would say whether the network earns its keep.

## License

MIT. See [LICENSE](LICENSE).
