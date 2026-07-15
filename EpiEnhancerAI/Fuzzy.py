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
Training = True

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


if str(Training) == "True":

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    output_dir = Path(sys.argv[3]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    columns_to_drop = ['seqnames', 'start', 'end', 'width', 'strand']
    X_train = train_data.drop(columns=columns_to_drop)
    X_test = test_data.drop(columns=columns_to_drop)

    y_train = train_data['STARR']
    y_test = test_data['STARR']

    class_names = np.unique(y_train)

    # Get class counts
    class_counts = np.bincount(y_train)
    minority_class = np.argmin(class_counts)
    majority_class = np.argmax(class_counts)

    print(f"Minority class: {minority_class} (n={class_counts[minority_class]})")
    print(f"Majority class: {majority_class} (n={class_counts[majority_class]})")

    
    print("filling NA")
    NA_num_code = -10
    # Compute the fuzzy partitions using n linguistic variables
    precomputed_partitions_vl = utils.construct_partitions(X_train, fz_type_studied, n_partitions=vl)

    precomputed_partitions_for_saving = copy.deepcopy(precomputed_partitions_vl)

    for i, v in enumerate(precomputed_partitions_vl):
        print(f'Variable {i}')
        if v.fuzzy_type() == fs.FUZZY_SETS.t1:
            NA_categori_FS = fs.categoricalFS('NA', NA_num_code)
        elif v.fuzzy_type() == fs.FUZZY_SETS.t2:
            NA_categori_FS = fs.categoricalIVFS('NA', NA_num_code)
        v.append(NA_categori_FS)
        for j, p in enumerate(v):
            print(f'Partition {j}')
            print(p)

    X_train = X_train.fillna(NA_num_code)
    X_test = X_test.fillna(NA_num_code)

    # ---------------------------------------------------------------
    # 80/20 split of X_train -> internal train / validation sets
    # (this is what the GA fits on and validates against)
    # ---------------------------------------------------------------
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train,
        test_size=0.20,
        stratify=y_train,
        random_state=42
    )

    print("Creating partitions")
    # We create a FRBC with the precomputed partitions and the specified fuzzy set type,
    fl_classifier = GA.BaseFuzzyRulesClassifier(nRules=nRules, linguistic_variables=precomputed_partitions_vl,
                                                 nAnts=nAnts, class_names=class_names, n_linguistic_variables=vl,
                                                 fuzzy_type=fz_type_studied, verbose=True, tolerance=tolerance,
                                                 runner=runner)

    print("fitting model")
    # fl_classifier.customized_loss(utils.mcc_loss) Use this to change the loss function, but be sure to look at the API first
    fl_classifier.fit(X_fit, y_fit, n_gen=n_gen, pop_size=n_pop, checkpoints=10, random_state=0)

    # print(vis_rules.rules_to_latex(fl_classifier.rule_base))
    str_rules = eval_tools.eval_fuzzy_model(fl_classifier, X_fit, y_fit, X_val, y_val,
                                             plot_rules=False, print_rules=True, plot_partitions=False,
                                             return_rules=True)

    rules_out_path = output_dir / 'Fuzzy_Rulebase.txt'
    with open(rules_out_path, 'w') as f:
        f.write(str_rules)

    print("Saving Done")
    partition_out_path = output_dir / 'Fuzzy_partition.txt'
    # Save the fuzzy partitions to a plain text file (using the clean, NA-free copy)
    with open(partition_out_path, 'w') as f:
        str_partitions = persistence.save_fuzzy_variables(precomputed_partitions_for_saving)
        f.write(str_partitions)

    y_pred = fl_classifier.predict(X_test)

    y_fit_pred_proba = fl_classifier.predict_proba(X_fit)
    y_val_pred_proba = fl_classifier.predict_proba(X_val)
    y_test_pred_proba = fl_classifier.predict_proba(X_test)

    proba_df = pd.DataFrame(y_test_pred_proba, columns=[f'Prob_{cls}' for cls in class_names])

    # Combine with predictions (reset indices so the concat aligns row-for-row)
    results_df = X_test.reset_index(drop=True).copy()
    results_df['Actual'] = y_test.reset_index(drop=True)
    results_df['Probabilities'] = y_pred
    results_df = pd.concat([results_df, proba_df.reset_index(drop=True)], axis=1)

    # Save to CSV
    path = output_dir / 'Type2_FLS_Predictions.csv'
    results_df.to_csv(path, index=False)
    print('Predictions saved successfully!')

    class_names_str = [str(cls) for cls in class_names]

    auc_train = compute_auc(y_fit, y_fit_pred_proba, class_names)
    auc_val = compute_auc(y_val, y_val_pred_proba, class_names)
    auc_test = compute_auc(y_test, y_test_pred_proba, class_names)

    print(f"AUC (Train):      {auc_train:.4f}")
    print(f"AUC (Validation):  {auc_val:.4f}")
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



    

