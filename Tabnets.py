import optuna
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, \
    average_precision_score

# Step 1: Load and preprocess the data
# dt = pd.read_excel(r'.\data1\crucial germs cirrhosis.xlsx', header=None)
dt = pd.read_excel(r'.\data2\wt2d.xlsx', header=None)
# Extract features (OTU counts) and labels (health status)
labels = dt.iloc[0, 1:].apply(lambda x: 0 if 'n' in str(x).lower() else 1).values  # Health status (0 or 1)
features = dt.iloc[1:, 1:].values.astype(float)  # OTU table (numeric values)

# Split the data into training, validation, and testing sets
X_train, X_unified, y_train, y_unified = train_test_split(features.T, labels, test_size=0.3, random_state=42)



def objective(trial):
    # Sample hyperparameters from the search space
    n_d = trial.suggest_int('n_d', 8, 128, step=8)  # Number of decision layers
    n_a = trial.suggest_int('n_a', 8, 128, step=8)  # Number of attention layers
    # n_steps = trial.suggest_int('n_steps', 8, 128, step=8)  # Number of steps
    # n_shared = trial.suggest_int('n_shared', 2, 8, step=2)
    # Create the TabNet model with the suggested hyperparameters
    clf = TabNetClassifier(
        n_d=n_d,
        n_a=n_a,
        # n_steps=n_steps,
        # n_shared= n_shared
    )

    # Train the model
    clf.fit(
        X_train, y_train,
        eval_set=[(X_unified, y_unified)],
        # verbose=0  # Silent mode during training
    )

    # Get predictions
    preds_proba = clf.predict_proba(X_unified)[:, 1]

    # Calculate the AUC score (higher is better)
    auc = roc_auc_score(y_unified, preds_proba)

    return auc  # Optuna will try to maximize the AUC score


# Create and run the Optuna study
study = optuna.create_study(direction='maximize')  # Maximize AUC
study.optimize(objective, n_trials=50)  # Run 50 trials
best_params = study.best_params
print(f"Best hyperparameters: {best_params}")
# Step 3: Train the TabNet model
NA = best_params['n_a']
ND = best_params['n_d']
# NS = best_params['n_steps']
# NH = best_params['n_shared']
clf = TabNetClassifier(n_a=NA, n_d=ND)
clf.fit(
    X_train, y_train,
    eval_set=[(X_unified, y_unified)],
    max_epochs=100, patience=15, batch_size=16, virtual_batch_size=8,
    num_workers=0, drop_last=False
)

# Get predictions on the unified set (both validation and test)
y_pred_proba = clf.predict_proba(X_unified)[:, 1]  # Predict probabilities for class 1

# Convert probabilities to binary predictions
y_pred = (y_pred_proba >= 0.5).astype(int)

# Calculate various metrics
accuracy = accuracy_score(y_unified, y_pred)
auc = roc_auc_score(y_unified, y_pred_proba)
precision = precision_score(y_unified, y_pred)
recall = recall_score(y_unified, y_pred)
f1 = f1_score(y_unified, y_pred)
mcc = matthews_corrcoef(y_unified, y_pred)
aupr = average_precision_score(y_unified, y_pred_proba)

# Print the results
print("\nFinal Test Metrics (Unified Validation + Test Set):")
print(f"Model accuracy: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"MCC: {mcc:.4f}")
print(f"AUPR: {aupr:.4f}")