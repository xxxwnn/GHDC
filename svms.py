import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef, precision_recall_curve, auc, \
    roc_auc_score
from sklearn.utils import shuffle
import optuna

# 加载数据
file_path = r'.\data2\wt2d.xlsx'
df = pd.read_excel(file_path, header=None)

X = df.iloc[1:, 1:]

y = df.iloc[0, 1:].apply(lambda x: 0 if 'n' in str(x).lower() else 1).tolist()
print(y)
X = X.T

X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def objective(trial):
    C = trial.suggest_loguniform("C", 0.001, 10)
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
    gamma = trial.suggest_loguniform("gamma", 0.0001, 1) if kernel != "linear" else "scale"

    svm_model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    y_prob = svm_model.predict_proba(X_test)[:, 1]  # 获取正类的概率
    auc1 = roc_auc_score(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    aupr = auc(recall, precision)

    print(f"Accuracy: {accuracy:.4f}, MCC: {mcc:.4f}, AUPR: {aupr:.4f}")
    return auc1


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

best_params = study.best_params
print("Best parameters found by Optuna:", best_params)

svm_best_model = SVC(
    C=best_params["C"],
    kernel=best_params["kernel"],
    gamma=best_params["gamma"] if best_params["kernel"] != "linear" else "scale",
    probability=True,
    random_state=42
)

svm_best_model.fit(X_train, y_train)
y_pred_best = svm_best_model.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
mcc_best = matthews_corrcoef(y_test, y_pred_best)

y_prob_best = svm_best_model.predict_proba(X_test)[:, 1]
precision_best, recall_best, _ = precision_recall_curve(y_test, y_prob_best)
aupr_best = auc(recall_best, precision_best)

print(f"Best Model Accuracy: {accuracy_best:.4f}")
print(f"Best Model MCC: {mcc_best:.4f}")
print(f"Best Model AUPR: {aupr_best:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred_best))

test_accuracies = [trial.value for trial in study.trials]
plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, marker='o')
plt.xlabel('Trial')
plt.ylabel('AUC')
plt.title('Optuna Trials - Accuracy')
plt.grid(True)
plt.show()
