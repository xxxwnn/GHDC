import torch
import torch.nn as nn


# class CNNModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layer1 = nn.Sequential(nn.Conv1d(1, 16, 5, padding=2, dilation=2),  ### 对齐d=1 p=0 kernelsize=5
#                                     nn.ReLU())
#         self.layer2 = nn.Sequential(nn.Conv1d(1, 16, 5, padding=4, dilation=3),
#                                     nn.ReLU())
#         self.layer3 = nn.Sequential(nn.Conv1d(1, 16, 5, padding=6, dilation=4),
#                                     nn.ReLU())
#         self.layer4 = nn.Sequential(nn.Conv1d(1, 16, 5, padding=0, dilation=1),
#                                     nn.ReLU())
#
#         self.flatten = nn.Flatten()
#         self.sigmoid = nn.Sigmoid()
#         # self.linear = nn.Sequential(nn.LazyLinear(128), nn.ReLU(), nn.Linear(128, 2), nn.Sigmoid())
#         self.linear = nn.Sequential(nn.LazyLinear(128), nn.ReLU(), nn.Linear(128, 2), nn.Sigmoid())
#
#     def forward(self, x):
#         x1 = self.layer1(x)
#         x2 = self.layer2(x)
#         x3 = self.layer3(x)
#         x4 = self.layer4(x)
#         x = torch.cat([x1, x2, x3, x4], dim=1)
#         x = self.flatten(x)
#         x = self.linear(x)
#         probs = self.sigmoid(x)
#         return probs
#         # return x

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv1d(1, 32, 2, padding=2, dilation=2),  ### 对齐d=1 p=0 kernelsize=5
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv1d(32, 64, 2, padding=4, dilation=4),
                                    nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv1d(64, 128, 2, padding=6, dilation=3),
                                    nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv1d(128, 512, 2, padding=0, dilation=1),
                                    nn.ReLU())
        # self.layer5 = nn.Sequential(nn.Conv1d(256, 512, 5, padding=0, dilation=1),
        #                             nn.ReLU())
        self.flatten = nn.Flatten()
        # self.linear = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 2), nn.Sigmoid())
        self.linear = nn.Sequential(nn.LazyLinear(512), nn.ReLU(), nn.Linear(512, 2), nn.Sigmoid())
        # self.linear = nn.Sequential(nn.Linear(2496, 128), nn.ReLU(), nn.Linear(128, 2), nn.Sigmoid())
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(x.shape)
        x1 = self.layer1(x)
        # print(x1.shape)
        x2 = self.layer2(x1)
        # print(x2.shape)
        x3 = self.layer3(x2)
        # # print(x3.shape)
        x4 = self.layer4(x3)
        # x5 = self.layer5(x4)
        # print(x4.shape)
        x = self.flatten(x4)
        # print(x.shape)
        logits = self.linear(x)
        probs = self.sigmoid(logits)
        return probs


# class CNNModel(nn.Module):
#     def __init__(self):  # 默认支持2通道输入
#         super().__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv1d(1, 16, kernel_size=5, padding=2, dilation=2),
#             nn.ReLU()
#         )
#         self.layer2 = nn.Sequential(
#             nn.Conv1d(16, 64, kernel_size=5, padding=4, dilation=4),
#             nn.ReLU()
#         )
#         self.layer3 = nn.Sequential(
#             nn.Conv1d(64, 512, kernel_size=5, padding=6, dilation=3),
#             nn.ReLU()
#         )
#         self.layer4 = nn.Sequential(
#             nn.Conv1d(512, 512, kernel_size=5, padding=0, dilation=1),
#             nn.ReLU()
#         )
#         self.flatten = nn.Flatten()
#
#         # Optional: Adjust linear layer input according to expected output of last conv layer
#         self.linear = nn.Sequential(
#             nn.LazyLinear(512),  # LazyLinear allows dynamic input size
#             nn.ReLU(),
#             nn.Linear(512, 2),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         x1 = self.layer1(x)
#         x2 = self.layer2(x1)
#         x3 = self.layer3(x2)
#         x4 = self.layer4(x3)
#
#         # Flatten the output for the linear layer
#         x = self.flatten(x4)
#
#         logits = self.linear(x)
#         probs = torch.sigmoid(logits)  # You can apply sigmoid here, but it's also in the linear layer
#         return probs