import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_curve,classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from joblib import dump
import matplotlib.pyplot as plt
import xgboost as xgb
from scipy.stats import uniform, randint
import sys,os
from joblib import load
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

Training = True


if str(Training) == 'True':
    print("full feature set model running... ")
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    output_dir = Path(sys.argv[3]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
     
    # Separate features and the target variable
    X_train = train_data.drop('STARR', axis=1)
    y_train = train_data['STARR']
    X_test = test_data.drop('STARR', axis=1)
    y_test = test_data['STARR']
    # Drop the specified columns
    columns_to_drop = ['seqnames', 'start', 'end', 'width', 'strand']
    X_train_dropped = X_train.drop(columns=columns_to_drop)
    X_test_dropped = X_test.drop(columns=columns_to_drop)
       

    # Fill NaN values with mean and dumy variable encoding (binary encoding) in the datasets
    X_train_nan_mean = X_train_dropped.fillna(X_train_dropped.mean())
    X_test_nan_mean = X_test_dropped.fillna(X_test_dropped.mean())

    #creating dumy coulmns
    df_train_nan = X_train_dropped.isna().astype(int)
    df_test_nan = X_test_dropped.isna().astype(int)

    #cancatenate dumy coulmns
    X_train_filled = pd.concat([X_train_nan_mean, df_train_nan.add_suffix('_is_nan')], axis=1)
    X_test_filled = pd.concat([X_test_nan_mean, df_test_nan.add_suffix('_is_nan')], axis=1)


    drop_columns =['ATAC_is_nan', 'CHG_is_nan', 'CHH_is_nan', 'CpG_is_nan']
    if list(X_train_filled.columns) == drop_columns:
        X_train_filled = X_train_filled.drop(columns=drop_columns)
        X_test_filled = X_test_filled.drop(columns=drop_columns)
    else:
        drop_columns =['ATAC_is_nan']
        X_train_filled = X_train_filled.drop(columns=drop_columns)
        X_test_filled = X_test_filled.drop(columns=drop_columns)

    X_train_balanced = X_train_filled
    y_train_balanced = y_train
   


    # -----------------------------------------------------
    # 80-20 Train-Validation Split
    # -----------------------------------------------------
    X_train_balanced, X_val, y_train_balanced, y_val = train_test_split(
        X_train_balanced,
        y_train_balanced,
        test_size=0.20,
        random_state=42,
        stratify=y_train_balanced
    )
    
    print("here in the training section")
    
    # Define the parameter grid for RandomizedSearchCV with adjusted ranges
    param_dist = {
        'n_estimators': randint(50, 500),  # Adjusted range
        'max_depth': randint(3, 20),  # Adjusted range
        'learning_rate': uniform(0.001, 0.2),  # Adjusted range
        'subsample': uniform(0.6, 0.4),  # Ensures values between 0.6 and 1.0
        'colsample_bytree': uniform(0.6, 0.4),  # Ensures values between 0.6 and 1.0
        'reg_alpha': uniform(0.0, 1.0),  # Adjusted range
        'reg_lambda': uniform(0.0, 1.0)  # Adjusted range
    }
    # Initialize the XGBoost model
    xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist, n_iter=100, cv=3, n_jobs=1, verbose=10, scoring='roc_auc')
    # Fit RandomizedSearchCV to the training data
    random_search.fit(X_train_balanced, y_train_balanced)
    # Best model from RandomizedSearchCV
    best_xgb_model = random_search.best_estimator_
    #print(f"Best parameters found: {random_search.best_params_}")
    # Save the best model
    dump(best_xgb_model, 'xgboost_model.joblib')
    print("Best XGBoost model saved successfully!")


    # -----------------------------------------------------
    # Training Performance
    # -----------------------------------------------------
    y_train_prob = best_xgb_model.predict_proba(X_train_balanced)[:, 1]
    train_auc = roc_auc_score(y_train_balanced, y_train_prob)
    
    print(f"Training ROC-AUC: {train_auc:.4f}")

    # -----------------------------------------------------
    # Validation Performance
    # -----------------------------------------------------
    y_val_prob = best_xgb_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_prob)

    print(f"Validation ROC-AUC: {val_auc:.4f}")




def save_roc_curve_data_with_predictions(model, X, y,original_data, filename, threshold_range=np.linspace(0.5, 1.0, num=500)):
    """
        Computes and saves the ROC curve data and predictions for a given model and dataset with specified threshold range.

        Parameters:
        - model: Trained model to be evaluated.
        - X: Feature matrix.
        - y: True labels.
        - original_data: Original dataframe containing 'chr', 'start', and 'end' columns.
        - filename: Base filename for saving the outputs.
        - threshold_range: Range of threshold values to consider for generating the ROC curve.

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
    roc_data_path = output_dir / "XGBoost_data_roc.csv"
    roc_data.to_csv(roc_data_path, index=False)
    print(f"ROC AUC: {roc_auc:.2f} saved to {filename.replace('.csv', 'roc.csv')}")

    # Save the prediction data
    pred_data_path = output_dir / "XGBoost_data_predictions.csv"
    pred_data.to_csv(pred_data_path, index=False)
    print(f"Predictions saved to {filename.replace('.csv', 'predictions.csv')}")


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
    results.to_csv("XGBoost_data_prediction_threshold.csv", index=False)

    return thresholds



# Compute and save ROC curve data for the best XGBoost model on test data with continuous higher threshold values
save_roc_curve_data_with_predictions(best_xgb_model, X_test_filled, y_test, test_data,'roc_curve_xgboost_test.csv', threshold_range=np.linspace(0.5, 1.0, num=500))
# Plot the ROC curve for the test data
y_prob_test = best_xgb_model.predict_proba(X_test_filled)[:, 1]
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
plt.savefig("Xgboost_ROC.png")

# Find the optimal threshold using Youden's J statistic
youden_index = tpr_test - fpr_test
optimal_idx = np.argmax(youden_index)
optimal_threshold = thresholds_test[optimal_idx]
print(f'Optimal Threshold: {optimal_threshold:.2f}')
# Apply the optimal threshold to make new class predictions
y_pred_optimal = (y_prob_test >= optimal_threshold).astype(int)
# Evaluate the model on the holdout (test) data with the optimal threshold
accuracy_optimal = accuracy_score(y_test, y_pred_optimal)
print(f'Accuracy of the XGBoost model on holdout data with optimal threshold: {accuracy_optimal}')
# Generate and print the classification report on the holdout (test) data with the optimal threshold
report_optimal = classification_report(y_test, y_pred_optimal, target_names=['0', '1'], digits=2,output_dict=True)
print("Classification Report on holdout data with optimal threshold:")
print(report_optimal)
# Convert to DataFrame
report_df = pd.DataFrame(report_optimal).transpose()

# Save to CSV inside output_dir
report_csv_path = output_dir / "XGBoost_classification_report_optimal.csv"
report_df.to_csv(report_csv_path)



