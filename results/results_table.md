| Model | RMSE | RMSE (true RUL ≤ 30) | MAE | NASA score | R² | Recall @30 | Precision @30 | Ranking AP | Ranking AUC |
|---|---|---|---|---|---|---|---|---|---|
| Random Forest | 14.29 | 5.62 | 11.32 | 294 | 0.882 | 0.72 | 0.95 | 0.976 | 0.993 |
| XGBoost | 13.86 | 5.63 | 10.65 | 274 | 0.889 | 0.76 | 0.95 | 0.988 | 0.996 |
| LSTM ensemble | 15.22 | 3.99 | 11.32 | 452 | 0.866 | 0.92 | 0.96 | 0.983 | 0.995 |
| LSTM single seed (mean ± std, n=3) | 15.29 ± 0.24 | - | - | 459 ± 22 | - | - | - | - | - |

- RMSE(LSTM ensemble) - RMSE(XGBoost) = +1.33 cycles, 95% bootstrap CI [-0.30, +3.06] (CI includes 0: not a clear difference)
- RMSE(Random Forest) - RMSE(XGBoost) = +0.42 cycles, 95% bootstrap CI [-0.66, +1.50] (CI includes 0: not a clear difference)
- RMSE(LSTM ensemble) - RMSE(Random Forest) = +0.91 cycles, 95% bootstrap CI [-0.63, +2.69] (CI includes 0: not a clear difference)

*Ranking AP/AUC: how well predicted RUL separates the 25 test engines that really fail within 30 cycles from the rest; threshold-free, so models that simply predict lower values get no advantage.*
