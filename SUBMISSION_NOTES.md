# Submission Notes: TriageML Risk Scoring

## Model Choice
I selected **XGBoost (XGBClassifier)** integrated into a scikit-learn preprocessing pipeline, transitioning from the initial Random Forest baseline. 
* **Handling Imbalance:** XGBoost's `scale_pos_weight` parameter directly compensates for class imbalance (~3.4x more low-risk than high-risk cases), enhancing minority class recall.
* **Early Stopping & Optimization:** Stratified 3-Fold Cross-Validation with early stopping (evaluating PR-AUC / average precision) was used to find the optimal ensemble size (averaging 165 trees), protecting against overfitting.
* **Clinical Triage Alignment:** A custom decision threshold of `0.4` was set to favor recall on the high-risk class, ensuring fewer false negatives in clinical triage.

## Final Metrics (Held-Out Test Set, Threshold = 0.4)
The final model evaluated on the 20% held-out test set achieved **95% overall accuracy**:

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **0 (Low Risk)** | 0.97 | 0.97 | 0.97 | 125 |
| **1 (High Risk)** | 0.89 | 0.89 | 0.89 | 36 |
| **Accuracy** | | | 0.95 | 161 |
| **Macro Avg** | 0.93 | 0.93 | 0.93 | 161 |

## Handling Messy Rows
The dataset (`patients.csv`) had several data quality issues which were handled before training:
1. **Biologically Impossible Values:** Age values $\le 0$ (5 rows) and resting blood pressure values $\le 0$ (2 rows) were masked to `NaN` during preprocessing so they could be properly imputed rather than distorting the distribution.
2. **Extreme Outliers:** Capped cholesterol at `Q3 + 3 * IQR` to minimize the influence of extreme anomalies.
3. **Missing Data Imputation:** Imputed missing continuous values (RBP, cholesterol, max heart rate) with training set means and categorical values (chest pain type) with the most frequent value.
4. **Leakage Prevention:** To avoid data leakage, all imputation and scaling steps were wrapped in a `Pipeline` and fit strictly on the training partition after the initial 80/20 train/test split.

## Future Improvements
* **Advanced Imputation:** Experiment with KNN or iterative imputation (MICE) instead of simple mean/mode.
* **Hyperparameter Tuning:** Run a randomized search over max depth, subsampling rates, and learning rate.
* **Feature Engineering:** Create clinically relevant interaction features (e.g., double product: systolic BP $\times$ max heart rate).

## Problems I faced:
* balance of datset that is the dataset was imbalanced that is the number of rows with risk_label 0 was greater than the number of rows with risk_label 1. 
