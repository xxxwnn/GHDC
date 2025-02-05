import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc as pr_auc, matthews_corrcoef, recall_score, \
    precision_score, f1_score

from objectives import AdaptiveFocalLoss

# 数据读取
# dt = pd.read_excel(r'.\data1\crucial germs cirrhosis.xlsx', header=None)
dt = pd.read_excel(r'.\data2\obesity.xlsx', header=None)
df1 = dt.iloc[:, :]

# 提取标签并进行编码
numeric_labels = df1.iloc[0, 1:].apply(lambda x: 0 if 'n' in str(x).lower() else 1).tolist()
# print(numeric_labels)
df1 = df1.iloc[1:, 1:]  # 删除第一行和第一列，保留数值特征
x = df1.values.astype(np.float32).T  # 转置为 [样本数, 特征数]
y = [int(label) for label in numeric_labels]  # 转换标签为整数列表

# 模型超参数

sequence_length = x.shape[1]  # 特征数即序列长度
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "")


# 自定义数据集
class MyDataset(Dataset):
    def __init__(self, data, label):
        self.x = torch.tensor(data, dtype=torch.float32)  # 转换为张量
        self.y = torch.tensor(label, dtype=torch.long)  # 标签为整数

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


# 数据加载
dataset = MyDataset(x, y)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
input_dim = sequence_length  # 输入特征数
num_classes = 2  # 假设二分类
train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=0.2,random_state=42)
dtrain = MyDataset(train_X, train_y)
dtest = MyDataset(test_X, test_y)
train_loader = DataLoader(dataset=dtrain, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
test_loader = DataLoader(dataset=dtest, batch_size=16, shuffle=False, num_workers=0, drop_last=False)


def eval(net, test_loader, device):
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            imgs = imgs.to(device)  # 确保输入数据在正确的设备上
            targets = targets.to(torch.long).to(device)
            outputs = net(imgs)  # 假设你的模型不需要targets作为输入
            _, predicted = torch.max(outputs.data, 1)

            # 保存每个 batch 的标签和预测值
            all_labels.extend(targets.cpu().tolist())
            all_predictions.extend(predicted.cpu().tolist())

            # 保存概率用于计算 roc_auc_score
            probabilities = torch.softmax(outputs, dim=1)
            if probabilities.shape[1] == 2:
                probabilities = probabilities[:, 1]  # 获取正类的概率
            all_probabilities.extend(probabilities.cpu().tolist())

            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    # 计算 ROC AUC
    rocauc1 = roc_auc_score(all_labels, all_probabilities)
    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probabilities)
    aupr = pr_auc(recall_curve, precision_curve)
    print('Test Accuracy: {}/{} ({:.0f}%)'.format(correct, total, 100. * correct / total))
    mcc = matthews_corrcoef(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions, average='binary')
    precision = precision_score(all_labels, all_predictions, average='binary',zero_division=1)
    macro_f1 = f1_score(all_labels, all_predictions, average='binary')

    return correct / total, rocauc1, recall, precision, mcc, aupr, macro_f1


# 定义模型
class TransformerModel(nn.Module):
    def __init__(self, embedding_dim, num_classes, nhead, num_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(1, embedding_dim)  # 将输入特征映射到嵌入维度
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, num_classes)  # 全连接层，分类输出

    def forward(self, x):
        # 输入形状: [batch_size, sequence_length]
        # print(x.shape)
        x = x.unsqueeze(-1)
        x = self.embedding(x)  # [batch_size, sequence_length, embedding_dim]
        # print(x.shape)
        x = x.permute(1, 0, 2)  # 转换为 [sequence_length, batch_size, embedding_dim] 以适配 Transformer
        # print(x.shape)
        x = self.transformer_encoder(x)  # 通过 Transformer 编码器
        x = x.mean(dim=0)  # 对序列长度维度取平均，形状为 [batch_size, embedding_dim]
        x = self.fc(x)  # 全连接层输出
        return x


def objective(trial):
    # 定义超参数搜索空间
    def eval_model(model, data_loader, criterion, device):
        model.eval()  # Set the model to evaluation mode
        total_loss = 0
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        all_probabilities = []
        auc_scores = []
        with torch.no_grad():
            for data, targets in data_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)

                total_loss += loss.item() * targets.size(0)  # Accumulate loss
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
                # outputs = model(imgs, targets)
                probabilities = torch.softmax(outputs, dim=1)[:, 1]
                all_labels.extend(targets.cpu().tolist())
                all_probabilities.extend(probabilities.cpu().tolist())
                all_predictions.extend(predicted.cpu().tolist())

        avg_loss = total_loss / total  # Calculate average loss
        accuracy = correct / total if total > 0 else 0

        auc = roc_auc_score(all_labels, all_probabilities)
        auc_scores.append(auc)
        if len(auc_scores) > 10:  # 滚动窗口
            auc_scores.pop(0)

        avg_auc = sum(auc_scores) / len(auc_scores)

        return avg_auc
        # return avg_loss, accuracy

    embedding_dim = trial.suggest_int('embedding_dim', 64, 512)
    nhead = trial.suggest_int('nhead', 2, 6)
    num_layers = trial.suggest_int('num_layers', 1, 4)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    Epoches = trial.suggest_int('Epoches', 10, 100)

    # 确保 embedding_dim 能够被 nhead 整除
    if embedding_dim % nhead != 0:
        embedding_dim = (embedding_dim // nhead) * nhead

        # Print for debugging purposes
    print(f'Adjusted embedding_dim: {embedding_dim}, nhead: {nhead}')

    model = TransformerModel(embedding_dim, num_classes, nhead, num_layers).to(device)

    # 定义损失函数和优化器
    criterion = AdaptiveFocalLoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练和验证模型
    model.train()
    for epoch in range(Epoches):  # 简单的示例，只训练10个epoch
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)  # 将输入数据移动到 GPU
            target = target.to(device)  # 将目标数据移动到 GPU
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # 在验证集上评估模型性能
    model.eval()
    val_loss = 0
    correct = 0
    # val_loss, val_accuracy = eval_model(model, test_loader, criterion, device)
    auc = eval_model(model, test_loader, criterion, device)
    # print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    return auc  # Return to be minimized


if __name__ == '__main__':
    study = optuna.create_study(direction ="maximize")
    study.optimize(objective, n_trials=10)
    print("Best parameters:", study.best_params)
    best_params = study.best_params
    embedding_dims = best_params["embedding_dim"]
    nhead = best_params["nhead"]
    if embedding_dims % nhead != 0:
        embedding_dims = (embedding_dims // nhead) * nhead
    num_layers = best_params["num_layers"]
    learning_rate = best_params["learning_rate"]
    epoches = best_params['Epoches']

    model = TransformerModel(embedding_dims, num_classes, nhead, num_layers).to(device)

    # 训练设置
    # criterion = AdaptiveFocalLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    AUC = []
    # test_accuracies = []
    F1_scores = []
    MCC_scores = []
    Recall_scores = []
    Precision_scores = []
    AUPR_scores = []
    train_losses = []

    # 训练循环
    for epoch in range(epoches):
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x).to(device)  # 前向传播
            loss = criterion(outputs, batch_y)  # 计算损失
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
        accuracy, rocauc, recall, precision, mcc, aupr, f1 = eval(model, test_loader, device)
        # test_accuracies.append(accuracy)
        AUC.append(AUC)
        MCC_scores.append(mcc)
        Recall_scores.append(recall)
        Precision_scores.append(precision)
        F1_scores.append(f1)
        AUPR_scores.append(aupr)
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    accuracy, rocauc, recall, precision, mcc, aupr, f1 = eval(model, test_loader, device)
    print(f"test accuracy: {accuracy:.4f}")
    print(f"auc: {rocauc:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 score: {f1:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"AUPR: {aupr:.4f}")
