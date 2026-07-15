import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from joblib import dump
import matplotlib.pyplot as plt
from scipy.stats import uniform
import sys, os
from pathlib import Path
from joblib import load
import warnings
from sklearn.metrics import roc_curve, precision_recall_curve


warnings.filterwarnings("ignore")

# Load training data
Training = False


if str(Training) == 'False':
    print("full feature set model running... ")
    test_file = sys.argv[1]
    output_dir = Path(sys.argv[2]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    test_data = pd.read_csv(test_file)
    model_file = sys.argv[3]
    # Separate features and the target variable
    X_test = test_data.drop('STARR', axis=1)
    y_test = test_data['STARR']
    # Drop the specified columns
    columns_to_drop = ['seqnames', 'start', 'end', 'width', 'strand']
    X_test_dropped = X_test.drop(columns=columns_to_drop)
       

    # Fill NaN values with mean and dumy variable encoding (binary encoding) in the datasets
    X_test_nan_mean = X_test_dropped.fillna(X_test_dropped.mean())

    #creating dumy coulmns
    df_test_nan = X_test_dropped.isna().astype(int)

    #cancatenate dumy coulmns
    X_test_filled = pd.concat([X_test_nan_mean, df_test_nan.add_suffix('_is_nan')], axis=1)

    drop_columns =['ATAC_is_nan', 'CHG_is_nan', 'CHH_is_nan', 'CpG_is_nan']
    if list(X_test_filled.columns) == drop_columns:
            
        X_test_filled = X_test_filled.drop(columns=drop_columns)
    else:
        drop_columns =['ATAC_is_nan']
            
        X_test_filled = X_test_filled.drop(columns=drop_columns)

    
    model_path = output_dir / model_file
    best_logreg_model=load(model_path)
    #print(best_logreg_model)


def save_roc_curve_data_with_predictions(model, X, y, original_data, filename):
    """
            Computes and saves the ROC curve data and predictions for a given model and dataset.

            Parameters:
            - model: Trained model to be evaluated.
            - X: Feature matrix.
            - y: True labels.
            - original_data: Original dataframe containing 'chr', 'start', and 'end' columns.
            - filename: Base filename for saving the outputs.

            Outputs:
            - Saves ROC curve data to a CSV file.
            - Saves prediction data to a CSV file.
    """
    # Predict probabilities for the positive class
    y_prob = model.predict_proba(X)[:, 1]

    # Compute False Positive Rate (FPR), True Positive Rate (TPR), and thresholds
    fpr, tpr, thresholds = roc_curve(y, y_prob)

    # Compute the Area Under the ROC Curve (AUC)
    roc_auc = roc_auc_score(y, y_prob)

    # Create a DataFrame to save the ROC curve data
    roc_data = pd.DataFrame({
                'False Positive Rate': fpr,
                'True Positive Rate': tpr,
                'Thresholds': thresholds
    })

    # Create a DataFrame to save the predictions
    pred_data = original_data[['seqnames', 'start', 'end']].copy()
    pred_data['Actual'] = y
    pred_data['Probabilities'] = y_prob

    # Ensure pred_data only contains the specified columns
    pred_data = pred_data[['seqnames', 'start', 'end', 'Actual', 'Probabilities']]

    # Save the ROC curve data
    roc_data_path = output_dir / "LogReg_data_roc.csv"
    roc_data.to_csv(roc_data_path, index=False)
    print("ROC AUC: {roc_auc:.2f} saved")

    # Save the prediction data
    pred_data_path = output_dir / "LogReg_data_predictions.csv"
    pred_data.to_csv(pred_data_path, index=False)
            
    print("Predictions file saved")

    y_true = y
    y_score = y_prob

    # ---- ROC / Youden's J  ---------------
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)

    J = tpr - fpr
    pos = np.argmax(J)                 
    best_thresh_ROC = roc_thresholds[pos]

    # ---- F-score maximization  -----
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)

    
    with np.errstate(divide="ignore", invalid="ignore"):
        fscore = 2 * precision * recall / (precision + recall)
    fscore = fscore[:-1]  

    max_fscore = np.nanmax(fscore)
    pos = np.nanargmax(fscore)         
    best_thresh_PR = pr_thresholds[pos]

    avg_thresh = (best_thresh_ROC + best_thresh_PR) / 2
    
    results = pd.DataFrame({
        "Threshold_ROC": [best_thresh_ROC],
        "Threshold_precision": [best_thresh_PR],
        "Average_ROC_precision": [avg_thresh],
    })
    results.to_csv("LogReg_data_prediction_threshold.csv", index=False)

    return thresholds

# Compute and save ROC curve data for the best Logistic Regression model on test data
save_roc_curve_data_with_predictions(best_logreg_model, X_test_filled, y_test, test_data,'roc_curve_log_reg_test.csv')
# Plot the ROC curve for the test data
y_prob_test = best_logreg_model.predict_proba(X_test_filled)[:, 1]
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_prob_test)
roc_auc_test = roc_auc_score(y_test, y_prob_test)

plt.figure()
plt.plot(fpr_test, tpr_test, color='darkorange', lw=2, label=f'Test ROC curve (area = {roc_auc_test:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Test Data')
plt.legend(loc='lower right')
plt.savefig("LR_wholedata_ROC.png")

# Find the optimal threshold using Youden's J statistic
youden_index = tpr_test - fpr_test
optimal_idx = np.argmax(youden_index)
optimal_threshold = thresholds_test[optimal_idx]
print(f'Optimal Threshold: {optimal_threshold:.2f}')
# Apply the optimal threshold to make new class predictions
y_pred_optimal = (y_prob_test >= optimal_threshold).astype(int)
# Evaluate the model on the holdout (test) data with the optimal threshold
accuracy_optimal = accuracy_score(y_test, y_pred_optimal)
print(f'Accuracy of the Logistic Regression model on holdout data with optimal threshold: {accuracy_optimal:.2f}')

#Generate and print the classification report on the holdout (test) data with the optimal threshold
report_optimal = classification_report(y_test, y_pred_optimal, target_names=['0', '1'], digits=2,output_dict=True)
print("Classification Report on holdout data with optimal threshold:")
print(report_optimal)
# Convert to DataFrame
report_df = pd.DataFrame(report_optimal).transpose()

# Save to CSV inside output_dir
report_csv_path = output_dir / "LogReg_classification_report_optimal.csv"
report_df.to_csv(report_csv_path)



