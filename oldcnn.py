import pandas as pd
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_curve, \
    matthews_corrcoef, recall_score, precision_score, f1_score, auc as pr_auc
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
from sklearn.preprocessing import normalize, OneHotEncoder
from sklearn.model_selection import train_test_split
from dataloader import x, y, MyDataset


# class MyDataset(Dataset):  # Inherit from Dataset
#     def __init__(self, data, label):
#         self.x = data
#         self.y = label
#
#     def __len__(self):  # Return the size of the entire dataset
#         return len(self.x)
#
#     def __getitem__(self, index):  # Return the image and label by index
#         return torch.tensor(self.x[index], dtype=torch.float32).unsqueeze(0), torch.tensor(self.y[index],
#                                                                                            dtype=torch.long)


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv1d(1, 16, 5, 1, 2),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv1d(16, 32, 5, 1, 2),
                                    nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv1d(32, 64, 5, 1, 2),
                                    nn.ReLU())
        # self.layer4 = nn.Sequential(nn.Conv1d(64, 128, 5, 1, 2),
        #                             nn.ReLU())
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Sequential(nn.LazyLinear(64), nn.ReLU(), nn.Linear(64, 2), nn.Sigmoid())

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        # x4 = self.layer4(x3)
        x = self.flatten(x3)
        x = self.linear(x)
        probs = self.sigmoid(x)
        return probs


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'best_model_cnn.pth'
train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=0)
dtrain = MyDataset(train_X, train_y)
dtest = MyDataset(test_X, test_y)
if __name__ == "__main__":
    # dt = pd.read_excel(r'C:\Users\xxwn\Desktop\bio\Gut_flora_v1\data2\cirrhosis.xlsx', header=None)
    # df1 = dt.iloc[:, 1:]
    # # print(df1.T)
    # onehot = OneHotEncoder(sparse_output=False)
    # label = onehot.fit_transform(df1.T[[0]]).tolist()
    # # print(data)
    # y = df1.iloc[0, :]
    # # print(y)
    # label = []
    # for i in y:
    #     if i[0] == 'n':
    #         label.append(0)
    #     else:
    #         label.append(1)
    # df1 = df1.iloc[1:, ]
    # y = [int(x) for x in label]
    # x = np.array(df1.values).transpose()
    # x = x.astype(np.float32)
    # indices = np.random.permutation(x.shape[1])  # x.shape[1] 是 x 的列数

    # 使用这些索引来打乱 x 的列
    # x = x[:, indices]
    # length = x.shape[1]  # Determine the length of the input for the Linear layer

    # Basic Params-----------------------------
    epoch = 40
    learning_rate = 0.0001
    batch_size_train = 16
    batch_size_test = 16
    momentum = 0.5

    train_loader = DataLoader(dataset=dtrain, batch_size=batch_size_train, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=dtest, batch_size=batch_size_test, shuffle=False, num_workers=4)

    net = CNNModel().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # Train---------------------------------
    writer = SummaryWriter(log_dir='logs/{}'.format(time.strftime('%Y%m%d-%H%M%S')))


    def train(epoch):
        global total_train_step
        total_train_step = 0
        total_loss = 0  # 用于计算每个epoch的平均loss
        batch_count = 0
        correct = 0
        total = 0

        # 在每个 epoch 开始时，随机交换特征顺序
        feature_indices = np.random.permutation(train_X.shape[1])  # 获取随机顺序的列索引
        train_X_shuffled = train_X[:, feature_indices]  # 按随机顺序重新排列特征

        # 重新创建 Dataset 和 DataLoader
        dtrain_shuffled = MyDataset(train_X_shuffled, train_y)
        train_loader = DataLoader(dataset=dtrain_shuffled, batch_size=batch_size_train, shuffle=True, num_workers=4)

        for data in train_loader:
            imgs, targets = data
            imgs = imgs.unsqueeze(1).to(device)
            targets = targets.to(torch.long).to(device)
            optimizer.zero_grad()

            outputs = net(imgs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1
            total += targets.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == targets).sum().item()

            if total_train_step % 200 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, total_train_step, len(dtrain_shuffled),
                    100. * total_train_step / len(dtrain_shuffled), loss.item()))

            writer.add_scalar('loss', loss.item(), total_train_step)
            total_train_step += 1

        print(correct / total)

        # 返回当前epoch的平均loss
        return total_loss / batch_count


    # Test---------------------------------

    def eval():
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        all_probabilities = []

        with torch.no_grad():
            for data in test_loader:
                imgs, targets = data
                # print(imgs.shape)
                imgs = imgs.unsqueeze(1).to(device)  # 增加一个通道维度
                targets = targets.to(torch.long).to(device)
                # outputs = net(imgs, targets)
                outputs = net(imgs)
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
                print('Result:{}, True:{}'.format(predicted.tolist(), targets.tolist()))

        # 计算 ROC AUC
        rocauc1 = roc_auc_score(all_labels, all_probabilities)
        precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probabilities)
        aupr = pr_auc(recall_curve, precision_curve)
        print('Test Accuracy: {}/{} ({:.0f}%)'.format(correct, total, 100. * correct / total))
        mcc = matthews_corrcoef(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions, average='binary')
        precision = precision_score(all_labels, all_predictions, average='binary')
        # 计算 F1 分数
        macro_f1 = f1_score(all_labels, all_predictions, average='binary')
        return correct / total, rocauc1, recall, precision, mcc, aupr, macro_f1


    total_train_step = 0

    AUC = []
    test_accuracies = []
    F1_scores = []
    MCC_scores = []
    Recall_scores = []
    Precision_scores = []
    AUPR_scores = []
    train_losses = []
    best_score = None
    patience = 25
    counter = 0
    # Run----------------------------------
    for i in range(1, epoch + 1):
        print(f"-----------------Epoch: {i}-----------------")

        # 每个epoch前打乱训练数据
        # train_X, train_y = shuffle(train_X, train_y, random_state=0)

        # 重新创建 Dataset 和 DataLoader
        # dtrain = MyDataset(train_X, train_y)
        train_loader = DataLoader(dataset=dtrain, batch_size=16, shuffle=True, num_workers=0,
                                  drop_last=True)
        # print(train_loader.shape)
        train(i)
        # test_accuracy= test()
        test_accuracy, auc, recall, precision, mcc, aupr, f1 = eval()
        test_accuracies.append(test_accuracy)
        AUC.append(auc)
        MCC_scores.append(mcc)
        Recall_scores.append(recall)
        Precision_scores.append(precision)
        F1_scores.append(f1)
        AUPR_scores.append(aupr)
        writer.add_scalar('test_accuracy', test_accuracy, total_train_step)

        torch.save(net.state_dict(), f'model/{model_path}')
        if best_score is None or test_accuracy >= best_score:  # 以 AUC 为例
            best_score = test_accuracy
            counter = 0
            torch.save(net.state_dict(), f'model/{model_path}')  # 保存最优模型
            print('Saved best model')
        else:
            counter += 1

            # 如果性能连续未提升超过 patience 次，停止训练
        if counter >= patience:
            print("Early stopping triggered!")
            break

    plt.plot(range(1, epoch + 1), test_accuracies, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy')
    plt.title('Training Accuracy over Epochs')
    plt.grid(True)
    plt.show()
