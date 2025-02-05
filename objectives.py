import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score, precision_recall_curve, \
    auc as pr_auc
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from GHDC import CNNModel
from dataloader import MyDataset, model_path, x, y

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(input, target)  # 计算交叉熵损失
        pt = torch.exp(-ce_loss)  # 计算概率
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss  # 计算Focal Loss
        # 确保 focal_loss 是标量
        if self.reduction == 'mean':
            return focal_loss.mean()  # 返回标量张量
        elif self.reduction == 'sum':
            return focal_loss.sum()  # 返回标量张量
        else:
            return focal_loss  # 如果不指定 reduction，则返回每个样本的损失张量


class AdaptiveFocalLoss(nn.Module):
    def __init__(self, initial_alpha=1, initial_gamma=2, decay_factor=0.99):
        super(AdaptiveFocalLoss, self).__init__()
        self.alpha = initial_alpha
        self.gamma = initial_gamma
        self.decay_factor = decay_factor

    def forward(self, input, target):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(input, target)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

    def decay_params(self):
        self.alpha *= self.decay_factor
        self.gamma *= self.decay_factor


def objective(trial):
    # 定义并记录初始超参数
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 3e-4)
    # batch_size_train = trial.suggest_categorical("batch_size_train", [8, 16, 32])
    # alpha = trial.suggest_float("alpha", 0.5, 2.0)
    # gamma = trial.suggest_float("gamma", 1, 5)
    num_epochs = trial.suggest_int("num_epochs", 300, 600)
    dilation1 = trial.suggest_int("dilation1", 1, 5)
    dilation2 = trial.suggest_int("dilation2", 1, 5)
    dilation3 = trial.suggest_int("dilation3", 1, 5)
    dilation4 = trial.suggest_int("dilation4", 1, 5)
    # dilation5 = trial.suggest_int("dilation5", 1, 5)
    patience = 100  # 早停的等待次数
    best_auc = 0
    epochs_no_improve = 0  # 用于计数没有提升的epoch数量

    # 数据加载和准备
    train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=0.1, stratify=y, random_state=42)
    # print(train_X, train_y)
    train_x, test_x, Train_y, Test_y = train_test_split(train_X, train_y, test_size=0.2, random_state=42)
    dtrain = MyDataset(train_x, Train_y)
    dtest = MyDataset(test_x, Test_y)
    train_loader = DataLoader(dataset=dtrain, batch_size=16, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=dtest, batch_size=16, shuffle=False, drop_last=False)

    # 初始化模型、损失函数、优化器和学习率调度器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel().to(device)
    model.layer1[0].dilation = (dilation1,)
    model.layer1[0].padding = (dilation1 * 2,)
    model.layer2[0].dilation = (dilation2,)
    model.layer2[0].padding = (dilation2 * 2,)
    model.layer3[0].dilation = (dilation3,)
    model.layer3[0].padding = (dilation3 * 2,)
    model.layer4[0].dilation = (dilation4,)
    model.layer4[0].padding = (dilation4 * 2)
    # model.layer5[0].dilation = (dilation5,)
    # model.layer5[0].padding = (dilation5 * 2)
    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.MSELoss()
    # loss_fn = AdaptiveFocalLoss(initial_alpha=alpha, initial_gamma=gamma).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # 训练和验证循环
    auc_scores = []
    for epoch in range(num_epochs):
        model.train()
        for data in train_loader:
            imgs, targets = data
            imgs = imgs.unsqueeze(1).to(device)
            targets = targets.to(torch.long).to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            # outputs = model(imgs, targets)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        all_labels = []
        all_probabilities = []
        all_predictions = []
        with torch.no_grad():
            for data in test_loader:
                imgs, targets = data
                imgs = imgs.unsqueeze(1).to(device)
                targets = targets.to(torch.long).to(device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs.data, 1)
                # outputs = model(imgs, targets)
                probabilities = torch.softmax(outputs, dim=1)[:, 1]
                all_labels.extend(targets.cpu().tolist())
                all_probabilities.extend(probabilities.cpu().tolist())
                all_predictions.extend(predicted.cpu().tolist())

            # 计算并记录AUC
            # macro_f1 = f1_score(all_labels, all_predictions, average='binary')
            auc = roc_auc_score(all_labels, all_probabilities)
            auc_scores.append(auc)
            if len(auc_scores) > 10:  # 滚动窗口
                auc_scores.pop(0)

            avg_auc = sum(auc_scores) / len(auc_scores)

            # 检查是否有提升
            if avg_auc > best_auc:
                best_auc = avg_auc
                torch.save(model.state_dict(), model_path)  # 保存最佳模型
                epochs_no_improve = 0  # 重置没有提升的epoch计数
            else:
                epochs_no_improve += 1  # 增加没有提升的epoch计数

            # 如果连续的epoch都没有提升AUC，提前停止
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

            # 调整学习率并打印
            scheduler.step(avg_auc)
            # print("Current Learning Rate:", scheduler.get_last_lr())

        # 每个epoch衰减Focal Loss的参数
        # loss_fn.decay_params()

    return best_auc  # 将最优AUC返回Optuna
