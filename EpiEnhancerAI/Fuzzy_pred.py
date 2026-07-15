#!python -m pip install git+https://github.com/Fuminides/ex-fuzzy
#!python -m pip install imbalanced-learn
import csv
import re
import copy
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import warnings
import sys, os
from pathlib import Path
 
# This is for launching from root folder path
sys.path.append('./ex_fuzzy/')
sys.path.append('./ex_fuzzy/ex_fuzzy/')

# This is for launching from Demos folder
sys.path.append('../ex_fuzzy/')
sys.path.append('../ex_fuzzy/ex_fuzzy/')

import ex_fuzzy.fuzzy_sets as fs
import ex_fuzzy.evolutionary_fit as GA
import ex_fuzzy.utils as utils
import ex_fuzzy.eval_tools as eval_tools
import ex_fuzzy.pattern_stability as pattern_stability
import ex_fuzzy.persistence as persistence
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, classification_report,precision_recall_curve

warnings.filterwarnings("ignore")

# Load training data
Training = False

runner = 1  # 1: single thread, 2+: corresponding multi-thread
# Choose the parameters for the genetic algorithm
n_gen = 200
n_pop = 200

# Max number of rules and max number of antecedents per rule
nRules = 100
nAnts = 3

# Number of linguistic variables to use in the fuzzy variables
vl = 3

# Tolerance for the dominance score for each rule
tolerance = 0.00005

# Fuzzy set type used throughout the script (was missing before -> NameError)
fz_type_studied = fs.FUZZY_SETS.t2

Training = False
def compute_auc(y_true, y_proba, class_names):
    n_classes = len(class_names)
    try:
        if n_classes == 2:
            # Use probability of the positive (second) class
            auc = roc_auc_score(y_true, y_proba[:, 1])
        else:
            # One-vs-rest, macro-averaged (robust to class imbalance-agnostic reporting)
            auc = roc_auc_score(
                y_true, y_proba,
                multi_class='ovr',
                average='macro',
                labels=class_names
            )
    except ValueError as e:
        print(f"AUC could not be computed: {e}")
        auc = np.nan
    return auc


    
if str(Training) == "False":

    test_file = sys.argv[1]
    output_dir = Path(sys.argv[2]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    test_data = pd.read_csv(test_file)

    columns_to_drop = ['seqnames', 'start', 'end', 'width', 'strand']

    X_test = test_data.drop(columns=columns_to_drop)

    y_test = test_data['STARR_seq_binary']

    class_names = np.unique(y_test)

    # Get class counts
    class_counts = np.bincount(y_test)
    minority_class = np.argmin(class_counts)
    majority_class = np.argmax(class_counts)

    print(f"Minority class: {minority_class} (n={class_counts[minority_class]})")
    print(f"Majority class: {majority_class} (n={class_counts[majority_class]})")

    partition_file = sys.argv[3]
    rule_file = sys.argv[4]

    
    # Separate minority and majority samples
    X_minority = X_test[y_test == minority_class]
    X_majority = X_test[y_test == majority_class]

    # Fit scaler only on minority class
    scaler = StandardScaler().fit(X_minority)
    # Transform all data using minority-class scaling
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame to maintain column names
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # Load the saved fuzzy partitions from a file
    partition_path = partition_file
    with open(partition_path, 'r') as f:
        precomputed_partitions = persistence.load_fuzzy_variables(f.read())

    rule_file_path = rule_file
    with open(rule_file_path, 'r') as f:
        str_rules = f.read()

    # Load precomputed membership and rules
    mrule_base = persistence.load_fuzzy_rules(str_rules, precomputed_partitions)

    # Train model
    fl_classifier = GA.BaseFuzzyRulesClassifier(precomputed_rules=mrule_base,
                                                 linguistic_variables=precomputed_partitions,
                                                 nAnts=nAnts,
                                                 class_names=class_names,
                                                 n_linguistic_variables=vl,
                                                 fuzzy_type=precomputed_partitions[0].fuzzy_type(),
                                                 verbose=True,
                                                 tolerance=tolerance,
                                                 runner=runner)
    fl_classifier.load_master_rule_base(mrule_base)

    # Impute the NaN values in the test data
    X_test = X_test.fillna(precomputed_partitions[0][-1].category)

    y_pred = fl_classifier.predict(X_test)
    y_pred_proba = fl_classifier.predict_proba(X_test)  # Shape: (n_samples, n_classes)

    
    y_pred_proba_avg = y_pred_proba if y_pred_proba.ndim == 2 else y_pred_proba.mean(axis=1)

    proba_df = pd.DataFrame(y_pred_proba_avg, columns=[f'Prob_{cls}' for cls in class_names])

    # Combine with predictions
    results_df = X_test.copy()
    results_df['Actual'] = y_test
    results_df['Probabilities'] = y_pred
    results_df = pd.concat([results_df, proba_df], axis=1)

    # Save to CSV
    path = output_dir / 'Type2_FLS_Predictions.csv'
    results_df.to_csv(path, index=False)
    print('Predictions saved successfully!')

    # Convert class names to strings if they aren't already
    class_names_str = [str(cls) for cls in class_names]

    auc_test = compute_auc(y_test, y_pred_proba_avg, class_names)

    print(f"AUC (Test):        {auc_test:.4f}")

    report_optimal = classification_report(y_test, y_pred, target_names=class_names_str, output_dict=True)
    print("Classification Report on data ")
    print(report_optimal)
    # Convert to DataFrame
    report_df = pd.DataFrame(report_optimal).transpose()

    # Save to CSV inside output_dir
    report_csv_path = output_dir / "Fuzzy_classification_report_optimal.csv"
    report_df.to_csv(report_csv_path)

    y_true = y_test
    y_score = y_pred

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
    results.to_csv("Fuzzy_data_prediction_threshold.csv", index=False)

    

