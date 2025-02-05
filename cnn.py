import time
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from optuna.samplers import TPESampler
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score, precision_recall_curve, \
    auc as pr_auc
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from GHDC import CNNModel
from dataloader import MyDataset, x, y, model_path
from objectives import objective, AdaptiveFocalLoss

start_time = time.time()
# IBD初始化的时候，我们的参数一般都是0均值的，因此开始的拟合y=Wx+b，基本过原点附近，如图b红色虚线。因此，网络需要经过多次学习才能逐步达到如紫色实线的拟合，即收敛的比较慢。如果我们对输入数据先作减均值操作，如图c
# ，显然可以加快学习。更进一步的，我们对数据再进行去相关操作，使得数据更加容易区分，这样又会加快训练，如图d。 通过把梯度映射到一个值大但次优的变化位置来阻止梯度过小变化。
device = torch.device("cuda")
print(torch.cuda.is_available())
# device = torch.device('cuda:1') #数字切换卡号
print(device)


# def train(epoch):
#     global total_train_step
#     total_train_step = 0
#     zero_value_counts = []
#     for data in train_loader:
#         # print(data)
#         imgs, targets = data
#         imgs = imgs.unsqueeze(1).to(device)
#         perm = torch.randperm(imgs.size(0))
#         imgs = imgs[perm]
#         targets = targets.to(torch.long).to(device)
#         targets = targets[perm]
#         optimizer.zero_grad()
#         # outputs = net(imgs, targets)
#         outputs = net(imgs)
#         loss = loss_fn(outputs, targets)
#         loss.backward()
#         optimizer.step()
#         if total_train_step % 200 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, total_train_step, len(dtrain),
#                 100. * total_train_step / len(dtrain), loss.item()))
#         writer.add_scalar('loss', loss.item(), total_train_step)
#         total_train_step += 1


def train(epoch):
    global total_train_step
    total_train_step = 0
    total_loss = 0  # 用于计算每个epoch的平均loss
    batch_count = 0
    correct = 0
    total = 0

    for data in train_loader:
        imgs, targets = data
        # print(imgs.shape)
        imgs = imgs.unsqueeze(1).to(device)
        # print(imgs.shape)
        targets = targets.to(torch.long).to(device)
        optimizer.zero_grad()
        # print(imgs.shape, targets.shape)
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
                epoch, total_train_step, len(dtrain),
                100. * total_train_step / len(dtrain), loss.item()))

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


if __name__ == '__main__':
    best_model = CNNModel().to(device)
    # 创建Optuna的study并启动优化
    study = optuna.create_study(direction="maximize", sampler=TPESampler())
    study.optimize(objective, n_trials=15)

    best_params = study.best_params
    # print(best_params['dilation1'], best_params['dilation2'], best_params['dilation3'], best_params['dilation4'])
    # best_model.layer1[0].dilation = (1,)
    # best_model.layer1[0].padding = (2,)
    best_model.layer1[0].dilation = (best_params['dilation1'],)
    best_model.layer1[0].padding = (best_params['dilation1'] * 2,)
    best_model.layer2[0].dilation = (best_params['dilation2'],)
    best_model.layer2[0].padding = (best_params['dilation2'] * 2,)
    best_model.layer3[0].dilation = (best_params['dilation3'],)
    best_model.layer3[0].padding = (best_params['dilation3'] * 2,)
    best_model.layer4[0].dilation = (best_params['dilation4'],)
    best_model.layer4[0].padding = (best_params['dilation4'] * 2)
    # best_model.layer5[0].dilation = (best_params['dilation5'],)
    # best_model.layer5[0].padding = (best_params['dilation5'] * 2)
    best_model.layer1[0].dilation = 1
    best_model.layer1[0].padding = 2
    best_model.layer2[0].dilation = 1
    best_model.layer2[0].padding = 2
    best_model.layer3[0].dilation = 1
    best_model.layer3[0].padding = 2
    best_model.layer4[0].dilation = 1
    best_model.layer4[0].padding = 2

    # Move model to GPU
    net = best_model
    # net = CNNModel().to(device)

    train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=0.2, stratify=y)

    dtrain = MyDataset(train_X, train_y)
    # print(train_X.shape)
    dtest = MyDataset(test_X, test_y)

    # Basic Params-----------------------------
    epoch = best_params["num_epochs"]
    learning_rate = best_params["learning_rate"]
    # batch_size_train = best_params["batch_size_train"]
    # epoch = 500
    # learning_rate = 0.0001
    batch_size_test = 16
    gpu = torch.cuda.is_available()
    momentum = 0.6

    train_loader = DataLoader(dataset=dtrain, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(dataset=dtest, batch_size=16, shuffle=False, num_workers=0, drop_last=False)

    writer = SummaryWriter(log_dir='logs/{}'.format(time.strftime('%Y%m%d-%H%M%S')))

    # loss_fn = AdaptiveFocalLoss()
    loss_fn = nn.CrossEntropyLoss().to(device)
    # loss_fn = nn.MSELoss().to(device)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    AUC = []
    test_accuracies = []
    F1_scores = []
    MCC_scores = []
    Recall_scores = []
    Precision_scores = []
    AUPR_scores = []
    train_losses = []
    patience = 25
    # Run----------------------------------
    best_score = None
    counter = 0
    # for i in range(1, epoch + 1):
    #     print(f"-----------------Epoch: {i}-----------------")
    #
    #     # 每个epoch前打乱训练数据
    #     # train_X, train_y = shuffle(train_X, train_y, random_state=0)
    #
    #     # 重新创建 Dataset 和 DataLoader
    #     # dtrain = MyDataset(train_X, train_y)
    #     train_loader = DataLoader(dataset=dtrain, batch_size=16, shuffle=True, num_workers=0,
    #                               drop_last=True)
    #     # print(train_loader.shape)
    #     train(i)
    #     # test_accuracy= test()
    #     test_accuracy, auc, recall, precision, mcc, aupr, f1 = eval()
    #     test_accuracies.append(test_accuracy)
    #     AUC.append(auc)
    #     MCC_scores.append(mcc)
    #     Recall_scores.append(recall)
    #     Precision_scores.append(precision)
    #     F1_scores.append(f1)
    #     AUPR_scores.append(aupr)
    #     writer.add_scalar('test_accuracy', test_accuracy, total_train_step)
    #
    #     torch.save(net.state_dict(), f'model/{model_path}')
    #     if best_score is None or test_accuracy >= best_score:  # 以 AUC 为例
    #         best_score = test_accuracy
    #         counter = 0
    #         torch.save(net.state_dict(), f'model/{model_path}')  # 保存最优模型
    #         print('Saved best model')
    #     else:
    #         counter += 1
    #
    #         # 如果性能连续未提升超过 patience 次，停止训练
    #     if counter >= patience:
    #         print("Early stopping triggered!")
    #         break
    # best_score = None
    # counter = 0
    for i in range(1, epoch + 1):
        print(f"-----------------Epoch: {i}-----------------")
        # 重新创建 Dataset 和 DataLoader
        # dtrain = MyDataset(train_X, train_y)
        # train_loader = DataLoader(dataset=dtrain, batch_size=16, shuffle=True, num_workers=0,
        #                           drop_last=True)
        train_loss = train(i)
        train_losses.append(train_loss)

        print(f"Epoch {i}, Train Loss: {train_loss:.6f}, Best Score: {best_score if best_score else 'None'}")

        if best_score is None or best_score - train_loss > 0.001:
            best_score = train_loss
            counter = 0
            torch.save(net.state_dict(), f'model/{model_path}')
            print('Saved best model based on loss')
        else:
            counter += 1
            print(f"Early stopping counter: {counter}/{patience}")
        if counter >= patience:
            print("Early stopping triggered due to loss plateau!")
            break

    # 训练完成后在最后进行评估
    print("Training completed. Evaluating the best model...")
    net.load_state_dict(torch.load(f'model/{model_path}', weights_only=True))  # 加载最优模型

    test_accuracy, auc, recall, precision, mcc, aupr, f1 = eval()
    

    print(f"test accuracy: {test_accuracy:.4f}")
    print(f"auc: {auc:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 score: {f1:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"AUPR: {aupr:.4f}")
    print(f"Final Test Accuracy: {test_accuracy}")

    # # 保存模型
    # torch.save(net.state_dict(), f'model/{model_path}')
    # print('Saved model')
    # print(f"test accuracy: {max(test_accuracies):.4f}")
    # print(f"auc: {max(AUC):.4f}")
    # print(f"Average Recall: {np.mean(Recall_scores):.4f}")
    # print(f"Average Precision: {np.mean(Precision_scores):.4f}")
    # print(f"Average F1 score: {np.mean(F1_scores):.4f}")
    # print(f"Average MCC: {np.mean(MCC_scores):.4f}")
    # print(f"Average AUPR: {np.mean(AUPR_scores):.4f}")
    # end_time = time.time()
    # total_runtime = end_time - start_time
    # print(f"Total runtime: {total_runtime // 3600:.0f}h {(total_runtime % 3600) // 60:.0f}m {total_runtime %
    # 60:.0f}s") print("Best hyperparameters:", study.best_params) print("Best AUC score:", study.best_value)

    # epochs_completed = len(test_accuracies)
    # plt.plot(range(1, epochs_completed + 1), test_accuracies, marker='o')
    # plt.xlabel('Epoch')
    # plt.ylabel('Training Accuracy')
    # plt.title('Training Accuracy over Epochs')
    # plt.grid(True)
    # plt.show()

    # epochs_completed = len(train_losses)
    # plt.plot(range(1, epochs_completed + 1), train_losses, marker='o')
    # plt.xlabel('Epoch')
    # plt.ylabel('Training Loss')
    # plt.title('Training Loss over Epochs')
    # plt.grid(True)
    # plt.show()

# dummy_input = torch.randn(16, 1, 47)
