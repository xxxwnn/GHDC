import numpy as np
import pandas as pd
import shap
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader

from cnn import MyDataset, CNNModel

model_path = r'C:\Users\xxwn\Desktop\bio\Gut_flora_v1\model\best_model2.pth'

# 加载模型
net = CNNModel()  # 初始化相同结构的模型
# net.load_state_dict(torch.load(fr"{model_path}"))
net.load_state_dict(torch.load('model/best_model2.pth'))

net.eval()  # 设置为评估模式
# dt = pd.read_excel(r'C:\Users\xxwn\Desktop\bio\Gut_flora_v1\SHAP_data\wt2d_processed.xlsx', header=None)
dt = pd.read_excel(r'C:\Users\xxwn\Desktop\bio\Gut_flora_v1\data1\crucial germs t2d.xlsx', header=None)
df1 = dt.iloc[:, :]
numeric_labels = df1.iloc[0, 1:].apply(lambda x: 0 if 'n' in str(x).lower() else 1).tolist()
# print(numeric_labels)
df1 = df1.iloc[1:, 1:]
x = df1.values.astype(np.float32).T
y = [int(label) for label in numeric_labels]

train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)
dtrain = MyDataset(train_X, train_y)
dtest = MyDataset(test_X, test_y)
# train_loader = DataLoader(dataset=dtrain, batch_size=batch_size_train, shuffle=True, num_workers=0, drop_last=True)
test_loader = DataLoader(dataset=dtest, batch_size=16, shuffle=False, num_workers=0, drop_last=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
species_names = dt.iloc[1:, 0].tolist()
background_size = 100  # 您可以根据需要调整样本数量
x = torch.from_numpy(x).float().unsqueeze(1)
# background = x.sample(n=background_size, random_state=42)  # 随机抽样
# background_np = background.to_numpy()  # 将背景数据集转换为NumPy数组
explainer = shap.DeepExplainer(net, x)
# 确保 SHAP 输入为正确形状
test_X_tensor = torch.tensor(test_X).float().unsqueeze(1)  # [47, 1, 50]
print("Input data shape (test_X_tensor):", test_X_tensor.shape)
masker = shap.maskers.Independent(test_X_tensor.detach().cpu().numpy())  # 使用 SHAP 内置的屏蔽器

# 计算 SHAP 值
shap_values = explainer(test_X_tensor)
# print(shap_values)
shap_values_class_1 = shap_values[..., 1]
shap_values_array_class_1 = shap_values_class_1.values  # 提取 SHAP 值为 numpy 数组
shap_values_mean = shap_values_array_class_1.mean(axis=0)
# 移除通道维度
shap_values_squeezed = shap_values_array_class_1.squeeze(1)
# print("Extracted SHAP values shape:", shap_values_squeezed.shape)
test_X_tensor_squeezed = test_X_tensor.squeeze(1)  # [47, 50]
# print(test_X_tensor_squeezed.shape)
assert shap_values_squeezed.shape == test_X_tensor_squeezed.shape, "Shape mismatch!"
# 可视化 SHAP 值（根据需要修改）
shap.summary_plot(shap_values_squeezed, test_X_tensor_squeezed, feature_names=species_names, show=False,max_display=10)  # 条形图)
# plt.gcf().set_size_inches(12, 8)  # 宽 12 高 8 手动显示
plt.tight_layout()
plt.show()
# index = 0
# test_X_numpy = test_X_tensor_squeezed.numpy()
# shap.dependence_plot(index, shap_values_squeezed, test_X_numpy, feature_names=species_names, show=False)
# plt.tight_layout()
# plt.show()
# shap.plots.bar(shap_values_mean, max_display=10)
# shap.decision_plot(explainer.expected_value[0], shap_values_squeezed, test_X_tensor_squeezed)
# shap.plots.beeswarm(shap_values_class_1)
