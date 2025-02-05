import numpy as np
import xgboost as xgb
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, \
    average_precision_score, log_loss
from torch import nn
from xgboost import XGBModel

dt = pd.read_excel(r'.\data2\t2d.xlsx', header=None)
# Extract features (OTU counts) and labels (health status)
labels = dt.iloc[0, 1:].apply(lambda x: 0 if 'n' in str(x).lower() else 1).values  # Health status (0 or 1)
features = dt.iloc[1:, 1:].values.astype(float)  # OTU table (numeric values)

# Split the data into training, validation, and testing sets
X_train, X_unified, y_train, y_unified = train_test_split(features.T, labels, test_size=0.2, random_state=42)
X_eval, X_test, y_eval, y_test = train_test_split(X_unified, y_unified, test_size=0.5, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_eval, label=y_eval)
dtest = xgb.DMatrix(X_test, label=y_test)


# 定义目标函数
def objective(trial):
    params = {
        'objective': 'binary:logistic',  # 二分类问题
        'eval_metric': 'auc',  # 使用AUC作为评价指标
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    }

    # 训练模型
    model = xgb.train(params, dtrain, num_boost_round=100, early_stopping_rounds=10,
                      evals=[(dtrain, 'train'), (dval, 'validation')])

    # 预测验证集
    y_pred_proba = model.predict(dval)

    # 计算AUC
    auc = roc_auc_score(y_eval, y_pred_proba)

    return auc


# 运行Optuna优化
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

# 获取最佳超参数
best_params = study.best_params
print(f"Best Params: {best_params}")

params = {
    'max_depth': best_params['max_depth'],  # 树的最大深度
    'eta': best_params['learning_rate'],  # 学习率
    'objective': 'binary:logistic',  # 二分类问题
    'n_estimators': best_params['n_estimators'],
    'subsample': best_params['subsample'],
    'colsample_bytree': best_params['colsample_bytree'],
    'min_child_weight': best_params['min_child_weight'],
}

xgb_model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtrain, 'train'), (dval, 'validation')],
                      early_stopping_rounds=10)

accuracies = []
aucs = []
precisions = []
recalls = []
F1 = []
mccs = []
auprs = []
for i in range(30):
    y_pred_proba = xgb_model.predict(dtest)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    # Convert probabilities to

# Calculate various metrics
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    auc = roc_auc_score(y_test, y_pred_proba)
    aucs.append(auc)
    precision = precision_score(y_test, y_pred)
    precisions.append(precision)
    recall = recall_score(y_test, y_pred)
    recalls.append(recall)
    f1 = f1_score(y_test, y_pred)
    F1.append(f1)
    mcc = matthews_corrcoef(y_test, y_pred)
    mccs.append(mcc)
    aupr = average_precision_score(y_test, y_pred_proba)
    auprs.append(aupr)


print("\nFinal Test Metrics (Unified Validation + Test Set):")
print(f"Model accuracy: {np.mean(accuracies):.4f}")
print(f"AUC: {np.mean(aucs):.4f}")
print(f"Precision: {np.mean(precisions):.4f}")
print(f"Recall: {np.mean(recalls):.4f}")
print(f"F1 Score: {np.mean(F1):.4f}")
print(f"MCC: {np.mean(mccs):.4f}")
print(f"AUPR: {np.mean(auprs):.4f}")
