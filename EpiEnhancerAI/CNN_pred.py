import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, classification_report,precision_recall_curve
from sklearn.model_selection import train_test_split,ParameterGrid
import sys,os
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Load training data
Training = False

class DeepSTARR(nn.Module):
    def __init__(self, num_filters, kernel_size, dropout_rate, num_conv_layers, num_fc_layers):
        super(DeepSTARR, self).__init__()
        layers = []
        in_channels = 1

        # Add convolutional layers
        for _ in range(num_conv_layers):
            layers.append(nn.Conv2d(in_channels, num_filters, kernel_size=(kernel_size, 1), stride=(1, 1), padding=(1, 0)))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))
            in_channels = num_filters
            
        self.conv_layers = nn.Sequential(*layers)

        # Calculate the output size after conv and pooling layers
        conv_output_size = num_features
        for _ in range(num_conv_layers):
            conv_output_size = (conv_output_size + 2 * 1 - kernel_size) // 1 + 1  # After conv
            conv_output_size = (conv_output_size - 2) // 2 + 1  # After pool
        self.fc_input_dim = num_filters * conv_output_size * 1  # Calculate fc input dim

        # Add fully connected layers
        fc_layers = []
        for _ in range(num_fc_layers - 1):
            fc_layers.append(nn.Linear(self.fc_input_dim if len(fc_layers) == 0 else 256, 256))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout_rate))
            
        fc_layers.append(nn.Linear(256, 1))
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc_layers(x)
        return torch.sigmoid(x)


if str(Training) == 'False':
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
    

    # Reshape data for CNN
    num_features = X_test_filled.shape[1]
    X_test_reshaped = X_test_filled.values.reshape(-1, 1, num_features, 1)

    # Convert data to PyTorch tensors
    X_test_tensor = torch.tensor(X_test_reshaped, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    # Create DataLoader
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

 
    model_path = output_dir / model_file
    model_checkpoint = torch.load(model_path)
    best_model = DeepSTARR(
        num_filters=model_checkpoint['best_params']['num_filters'],
        kernel_size=model_checkpoint['best_params']['kernel_size'],
        dropout_rate=model_checkpoint['best_params']['dropout_rate'],
        num_conv_layers=model_checkpoint['best_params']['num_conv_layers'],
        num_fc_layers=model_checkpoint['best_params']['num_fc_layers']
    )
    best_model.load_state_dict(model_checkpoint['model_state_dict'])
    print("CNN model loaded successfully!")


# Evaluate the model
best_model.eval()
    
y_true = []
y_prob = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = best_model(inputs).squeeze()
        probabilities = torch.sigmoid(outputs)
        y_true.extend(labels.cpu().numpy())
        y_prob.extend(probabilities.cpu().numpy())

y_true = np.array(y_true)
y_prob = np.array(y_prob)
print("Done evaluation")
pred_data = pd.DataFrame({
            'Actual': y_true,
            'Probabilities': y_prob
})

pred_data_path = output_dir / "CNN_data_predictions.csv"
pred_data.to_csv(pred_data_path, index=False)
    

# Compute False Positive Rate (FPR), True Positive Rate (TPR), and thresholds
fpr, tpr, thresholds = roc_curve(y_true, y_prob)

# Compute the Area Under the ROC Curve (AUC)
    
roc_data = pd.DataFrame({
            'False Positive Rate': fpr,
            'True Positive Rate': tpr,
            'Thresholds': thresholds
})

# Create a DataFrame to save the predictions
roc_data_path = output_dir / "CNN_data_roc.csv"
roc_data.to_csv(roc_data_path, index=False)
    

y_true = y_true
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
results.to_csv("CNN_data_prediction_threshold.csv", index=False)

# Plot the ROC curve for the test data
fpr_test, tpr_test, thresholds_test = roc_curve(y_true, y_prob)
roc_auc_test = roc_auc_score(y_true, y_prob)
plt.figure()
plt.plot(fpr_test, tpr_test, color='darkorange', lw=2, label=f'Test ROC curve (area = {roc_auc_test:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Test Data')
plt.legend(loc='lower right')
roc_plot_path = output_dir / "CNN_ROC.png"
plt.savefig(roc_plot_path)

# Find the optimal threshold using Youden's J statistic
youden_index = tpr_test - fpr_test
optimal_idx = np.argmax(youden_index)
optimal_threshold = thresholds_test[optimal_idx]
print(f'Optimal Threshold: {optimal_threshold:.2f}')
# Apply the optimal threshold to make new class predictions
y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
# Evaluate the model on the holdout (test) data with the optimal threshold
accuracy_optimal = accuracy_score(y_true, y_pred_optimal)
print(f'Accuracy of the DeepSTARR model on holdout data with optimal threshold: {accuracy_optimal:.2f}')
# Generate and print the classification report on the holdout (test) data with the optimal threshold
report_optimal = classification_report(y_true, y_pred_optimal, target_names=['0', '1'], digits=2,output_dict=True)
print("Classification Report on holdout data with optimal threshold:")
print(report_optimal)
# Convert to DataFrame
report_df = pd.DataFrame(report_optimal).transpose()

# Save to CSV inside output_dir
report_csv_path = output_dir / "CNN_classification_report_optimal.csv"
report_df.to_csv(report_csv_path)

