import random
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def read_csv(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(file_path)
    target = df['target']
    featrues = df.drop(columns=['target'])
    return featrues.values,target.values

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, 5)
        )
    def forward(self, x):
        return self.model(x)
def train_nn(X_train: torch.Tensor, y_train: torch.Tensor) -> Classifier:
    model = Classifier()  # 初始化模型
    
    criterion = nn.CrossEntropyLoss()  # 定义损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # 定义优化器
    
    epochs = 150
    for epoch in range(epochs):
        # 前向传播
        outputs = model(X_train)  # 注意是X_train不是feature
        loss = criterion(outputs, y_train)  # 计算损失
        
        # 反向传播和优化
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        
        # # 打印训练信息
        # if (epoch+1) % 10 == 0:  # 每10个epoch打印一次
        #     print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return model

def save_model(model: nn.Module, file_path: str) -> None:
    #TODO
    torch.save(model.state_dict(),file_path)

def main() -> None:

    features, targets = read_csv('train_data.csv')
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # print(X_test_tensor,X_test_tensor.shape)
    # print(y_train_tensor,y_train_tensor.shape)
    model = train_nn(X_train_tensor, y_train_tensor)
    save_model(model, 'model.pth')


if __name__ == '__main__':
    main()