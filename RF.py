from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef, precision_recall_curve, auc, \
    roc_auc_score
from sklearn.utils import shuffle
import optuna

from dataloader import x, y

# 加载数据
# file_path = r'.\data2\cirrhosis.xlsx'
# df = pd.read_excel(file_path, header=None)
#
# X = df.iloc[1:, 1:]
# # y = df.iloc[0, 1:].apply(lambda x: 0 if 'n' == str(x).lower() else 1).tolist()
#
# y = df.iloc[0, 1:].apply(lambda x: 0 if 'n' in str(x).lower() else 1).tolist()
# print(X)
# X = X.T
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


def objective(trial):
    X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train, random_state=trial.number)
    # Optuna 参数调优
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 5, 30)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)

    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )

    rf_model.fit(X_train_shuffled, y_train_shuffled)
    y_pred = rf_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    y_prob = rf_model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    auc1 = roc_auc_score(y_test, y_prob)
    aupr = auc(recall, precision)

    print(f"Trial {trial.number}: Accuracy: {accuracy:.4f}, MCC: {mcc:.4f}, AUPR: {aupr:.4f}")
    return auc1


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

best_params = study.best_params
print("Best parameters found by Optuna:", best_params)

rf_best_model = RandomForestClassifier(
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    min_samples_split=best_params["min_samples_split"],
    random_state=42
)

rf_best_model.fit(X_train, y_train)
y_pred_best = rf_best_model.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
mcc_best = matthews_corrcoef(y_test, y_pred_best)

y_prob_best = rf_best_model.predict_proba(X_test)[:, 1]
precision_best, recall_best, _ = precision_recall_curve(y_test, y_prob_best)
aupr_best = auc(recall_best, precision_best)

print(f"Best Model Accuracy: {accuracy_best:.4f}")
print(f"Best Model MCC: {mcc_best:.4f}")
print(f"Best Model AUPR: {aupr_best:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred_best))

test_accuracies = [trial.value for trial in study.trials]
plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, marker='o')
plt.xlabel('Trial')
plt.ylabel('Accuracy')
plt.title('Optuna Trials - Accuracy')
plt.grid(True)
plt.show()
