import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


# dt = pd.read_csv(r'.\data\cirrhosis.csv', header=None)
# dt = pd.read_excel(r'.\data2\obesity.xlsx', header=None)
dt = pd.read_excel(r'.\data1\crucial germs wt2d.xlsx', header=None)
# dt = pd.read_csv(r'C:\Users\xxwn\Desktop\bio\Gut_flora_v1\新CG\crucial_germs_cirrhosis.csv', header=None)

df1 = dt.iloc[:, :]

# Extract numeric labels (0 and 1)
# numeric_labels = df1.iloc[0, :].apply(lambda x: 0 if x[0] == 'n' else 1).tolist()
numeric_labels = df1.iloc[0, 1:].apply(lambda x: 0 if 'n' in str(x).lower() else 1).tolist()
# numeric_labels = dt.iloc[0, 1:].apply(lambda x: 1 if str(x).lower() == 'cancer' else 0).tolist()
# print(numeric_labels)
# Drop the first row to keep only the features
df1 = df1.iloc[1:, 1:]
# print(df1)
# Convert features to NumPy array and transpose
x = df1.values.astype(np.float32).T
# indices = np.random.permutation(x.shape[1])  # x.shape[1] 是 x 的列数

# 使用这些索引来打乱 x 的列
# x = x[:, indices]
# z = df1.values.astype(np.float32)
# n_features = z.shape[0]
# indices = np.arange(n_features).astype(np.float32)  # Create an index for each feature
# indices = [x / 10000 for x in range(0, n_features)]
# Expand dimensions of x and indices to enable concatenation
# Reshape x to (number_of_samples, n_features, 1)
# x_expanded = z[np.newaxis, ...]  # (1, number_of_features, number_of_samples)
# print(x_expanded.shape)
# Reshape indices to (number_of_samples, n_features, 1)
# indices_expanded = np.expand_dims(indices, axis=1)  # (n_features, 1)
# indices_repeated = np.repeat(indices_expanded, x.shape[0], axis=1)  # (n_features, number_of_samples)
# indices_repeated = indices_repeated[np.newaxis, ...]  # (1, n_features, number_of_samples)

# print(indices_repeated.shape)
# Concatenate along the first axis to combine features and indices
# combined_data = np.concatenate((x_expanded, indices_repeated), axis=0)  # (2, n_features, number_of_samples)

# print(combined_data)
# Convert combined data to tensor
# combined_data_tensor = torch.tensor(combined_data, dtype=torch.float32)  # (2, n_features, number_of_samples)
# print(combined_data_tensor)
# combined_data_tensor.to_csv('combined.csv')
# df1 = pd.DataFrame(combined_data_tensor[0])
# df2 = pd.DataFrame(combined_data_tensor[1])
# df1.to_csv(r'combined.csv', index=False)
# df2.to_csv(r'combined2.csv', index=False)
# print(combined_data.shape)

# print(x)
# Convert numeric labels to a list of integers
y = [int(label) for label in numeric_labels]
model_path = 'best_model2.pth'
# print(y)

class MyDataset(Dataset):  # 继承Dataset
    def __init__(self, data, label):
        self.x = torch.tensor(data)
        self.y = torch.tensor(label)

    def __len__(self):  # 返回整个数据集的大小
        return len(self.x)

    def __getitem__(self, index):  # 根据索引index返回图像及标签
        return self.x[index], self.y[index]
