import pandas as pd
import numpy as np

# 读取数据
cir = pd.read_csv(r'C:\Users\xxwn\Desktop\bio\Gut_flora_v1\data1\obesity.csv', header=None)

# 记录需要删除的行索引
rows_to_drop = []
r = cir.shape[1]
print(r)
for index, row in cir.iterrows():
    a = 0
    for i in range(0, r):  # 遍历每一行的第1到第232列
        if row[i] != 0 and row[i] != '0':
            a += 1
    if a < r*0.15:
        rows_to_drop.append(index)  # 记录需要删除的行索引

# 在循环外一次性删除标记的行
cir = cir.drop(rows_to_drop, axis=0)
cir.to_excel(r"C:\Users\xxwn\Desktop\bio\Gut_flora_v1\data2\obesity.xlsx", header=None)
# 输出结果
print(cir)
