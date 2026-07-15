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
Training = True

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

if str(Training) == 'True':
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
    columns_to_drop = columns_to_drop = ['seqnames', 'start', 'end', 'width', 'strand']
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
    

    # Reshape data for CNN
    num_features = X_train_balanced.shape[1]
    X_train_reshaped = X_train_balanced.values.reshape(-1, 1, num_features, 1)
    X_val_reshaped = X_val.values.reshape(-1, 1, num_features, 1)
    X_test_reshaped = X_test_filled.values.reshape(-1, 1, num_features, 1)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_reshaped, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_balanced.values, dtype=torch.long)

    X_val_tensor = torch.tensor(X_val_reshaped, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)

    X_test_tensor = torch.tensor(X_test_reshaped, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print("Training start")
    
        # -----------------------------------------------------
        # AUC function
        # -----------------------------------------------------
    def compute_auc(model, loader):
        model.eval()
        probs, labels = [], []

        with torch.no_grad():
            for x, y in loader:
                out = torch.sigmoid(model(x))
                probs.extend(out.cpu().numpy())
                labels.extend(y.cpu().numpy())

        return roc_auc_score(labels, probs)


    # Define a reduced grid of hyperparameters to result in 16 combinations
    param_grid = {
        'num_filters': [32, 64],  # 2 values
        'kernel_size': [1, 2],  # 2 values
        'dropout_rate': [0.2, 0.3],  # 2 values
        'learning_rate': [1e-4, 1e-3],  # 2 values
        'num_conv_layers': [2],  # 1 value (fixed to reduce complexity)
        'num_fc_layers': [2, 3]  # 2 values
    }

    def train_model(params, patience=3):
        model = DeepSTARR(
            params['num_filters'],
            params['kernel_size'],
            params['dropout_rate'],
            params['num_conv_layers'],
            params['num_fc_layers'])
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        
        n_epochs = 10
        early_stop_patience = patience
        best_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(n_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            
            model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs).squeeze()
                    loss = criterion(outputs, labels.float())
                    val_running_loss += loss.item() * inputs.size(0)
            val_loss = val_running_loss / len(test_loader.dataset)

            if val_loss < best_loss:
                best_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stop_patience:
                print(f'Early stopping triggered at epoch {epoch + 1}')
                break

        return best_loss, model

    # Grid search
    best_params = None
    best_loss = float('inf')
    best_model = None
    best_trial = 0
    trial = 0

    # Start grid search

    for params in ParameterGrid(param_grid):
        print(f"Training with parameters: {params}")
        trial_loss, model = train_model(params)

        trial += 1

        if trial_loss < best_loss:
            best_loss = trial_loss
            best_params = params
            best_model = model
            best_trial = trial
        
        print(f"Trial {trial} finished with value: {trial_loss} and parameters: {params}. Best is trial {best_trial} with value: {best_loss}.")

    num_combinations = len(ParameterGrid(param_grid))
    print(f"Total number of parameter combinations: {num_combinations}")

    print('Best hyperparameters found: ', best_params)
    # -----------------------------------------------------
    # Final Evaluation (AUC)
    # -----------------------------------------------------
    train_auc = compute_auc(best_model, train_loader)
    val_auc = compute_auc(best_model, val_loader)
    

    print("\nAUC RESULTS")
    print("Train AUC:", train_auc)
    print("Val AUC:", val_auc)

    # Train the final model with the best hyperparameters
    criterion = nn.BCELoss()
    optimizer = optim.Adam(best_model.parameters(), lr=best_params['learning_rate'])
    n_epochs = 20
    train_losses = []
    test_losses = []
    for epoch in range(n_epochs):
        best_model.train()
        running_loss = 0.0
        batch_count = 0  # For additional batch-level logging
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = best_model(inputs).squeeze()
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
            # Batch-level logging for debugging
            batch_count += 1
            if batch_count % 10 == 0:  # Log every 10 batches
                print(f'Epoch {epoch + 1}, Batch {batch_count}: Batch loss: {loss.item()}')

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validate the model
        best_model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = best_model(inputs).squeeze()
                loss = criterion(outputs, labels.float())
                running_loss += loss.item() * inputs.size(0)
        test_loss = running_loss / len(test_loader.dataset)
        test_losses.append(test_loss)

        print(f'Epoch {epoch + 1}/{n_epochs}.. Train loss: {train_loss:.4f}.. Test loss: {test_loss:.4f}')
    

    # Save the model
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'test_losses': test_losses,
        'best_params': best_params
    }, 'cnn_model.pth')
    print("CNN model saved successfully!")


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

