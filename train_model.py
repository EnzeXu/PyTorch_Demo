import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

# 检查是否支持MPS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

# 加载数据
# data = pd.read_csv('mdmom_forNN_3July2024.csv')
data = pd.read_csv('small.csv')

# 获取所有列名
all_columns = data.columns.tolist()

# 定义数值变量和类别变量
numerical_vars = [
    "0", "1", "2", "3", "4", "5", "6", "7",
    "8", "9", "10"
]

# 将'SMM'列作为目标变量，其余的都是特征变量
X = data.drop('SMM', axis=1).values
y = data['SMM'].values

# 获取类别型和数值型变量的索引
categorical_indices = [i for i, col in enumerate(all_columns) if col not in numerical_vars and col != 'SMM']
numerical_indices = [i for i, col in enumerate(all_columns) if col in numerical_vars]

# print(f"categorical_indices: {categorical_indices}")
# print(f"numerical_indices: {numerical_indices}")

# 分离类别型和数值型变量
X_categorical = X[:, categorical_indices]
X_numerical = X[:, numerical_indices]

# 标准化数值型特征数据
scaler = StandardScaler()
X_numerical = scaler.fit_transform(X_numerical)

# 分割数据集为训练集、验证集和测试集
X_temp_categorical, X_test_categorical, X_temp_numerical, X_test_numerical, y_temp, y_test = train_test_split(
    X_categorical, X_numerical, y, test_size=0.2, random_state=42, stratify=y)
X_train_categorical, X_val_categorical, X_train_numerical, X_val_numerical, y_train, y_val = train_test_split(
    X_temp_categorical, X_temp_numerical, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

# 转换为 PyTorch 张量
X_train_categorical_tensor = torch.tensor(X_train_categorical, dtype=torch.long).to(device)
X_val_categorical_tensor = torch.tensor(X_val_categorical, dtype=torch.long).to(device)
X_test_categorical_tensor = torch.tensor(X_test_categorical, dtype=torch.long).to(device)
X_train_numerical_tensor = torch.tensor(X_train_numerical, dtype=torch.float32).to(device)
X_val_numerical_tensor = torch.tensor(X_val_numerical, dtype=torch.float32).to(device)
X_test_numerical_tensor = torch.tensor(X_test_numerical, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

# 创建数据加载器，减少批次大小
train_dataset = TensorDataset(X_train_categorical_tensor, X_train_numerical_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_categorical_tensor, X_val_numerical_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_categorical_tensor, X_test_numerical_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 构建 MLP 模型
class MLPWithEmbeddingAndNumerical(nn.Module):
    def __init__(self, categorical_input_dim, numerical_input_dim, embedding_dim):
        super(MLPWithEmbeddingAndNumerical, self).__init__()
        self.embedding = nn.Embedding(categorical_input_dim, embedding_dim)
        self.categorical_layer = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.numerical_layer = nn.Sequential(
            nn.Linear(numerical_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.combine_layer = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x_categorical, x_numerical):
        x_cat = self.embedding(x_categorical).mean(dim=1)
        x_cat = self.categorical_layer(x_cat)
        x_num = self.numerical_layer(x_numerical)
        x = torch.cat((x_cat, x_num), dim=1)
        x = self.combine_layer(x)
        return x

# 初始化模型、损失函数和优化器
categorical_input_dim = len(categorical_indices)
numerical_input_dim = len(numerical_indices)
embedding_dim = 16

model = MLPWithEmbeddingAndNumerical(categorical_input_dim, numerical_input_dim, embedding_dim).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)

# 定义 _save_checkpoint 函数
def _save_checkpoint(ckpt_file_path, model, epoch, global_step, optimizer):
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, ckpt_file_path)

# 清空之前的损失记录
train_losses = []
val_losses = []

# 训练模型
num_epochs = 100000
ckp_path = 'checkpoints'  # 定义 checkpoint 文件保存的目录
os.makedirs(ckp_path, exist_ok=True)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    global_step = epoch + 1  # 可以根据具体情况定义 global_step

    for X_batch_categorical, X_batch_numerical, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch_categorical, X_batch_numerical)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch_categorical, X_batch_numerical, y_batch in val_loader:
            outputs = model(X_batch_categorical, X_batch_numerical)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    scheduler.step()

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, lr: {optimizer.param_groups[0]["lr"]}')

        # 保存 checkpoint
        # print("*** save checkpoint ****")
        ckpt_file_path = os.path.join(ckp_path, f'step_{global_step}.pt')
        _save_checkpoint(ckpt_file_path, model, epoch, global_step, optimizer)

# 保存模型
torch.save(model.state_dict(), 'mlp_model_with_embedding_and_numerical.pth')

# 保存损失值到文本文件
with open('losses_embedding_numerical.txt', 'w') as f:
    for t_loss, v_loss in zip(train_losses, val_losses):
        f.write(f'{t_loss},{v_loss}\n')

# 评估模型
model.eval()
with torch.no_grad():
    y_pred_train = model(X_train_categorical_tensor, X_train_numerical_tensor).round()
    y_pred_val = model(X_val_categorical_tensor, X_val_numerical_tensor).round()
    y_pred_test = model(X_test_categorical_tensor, X_test_numerical_tensor).round()
    y_pred_test_prob = model(X_test_categorical_tensor, X_test_numerical_tensor)
    
    train_accuracy = (y_pred_train.eq(y_train_tensor).sum() / float(y_train_tensor.shape[0])).item()
    val_accuracy = (y_pred_val.eq(y_val_tensor).sum() / float(y_val_tensor.shape[0])).item()
    test_accuracy = (y_pred_test.eq(y_test_tensor).sum() / float(y_test_tensor.shape[0])).item()
    
    tn, fp, fn, tp = confusion_matrix(y_test_tensor.cpu(), y_pred_test.cpu()).ravel()
    precision = precision_score(y_test_tensor.cpu(), y_pred_test.cpu())
    recall = recall_score(y_test_tensor.cpu(), y_pred_test.cpu())
    auc = roc_auc_score(y_test_tensor.cpu(), y_pred_test_prob.cpu())
    
    print(f'Train Accuracy: {train_accuracy:.4f}')
    print(f'Validation Accuracy: {val_accuracy:.4f}')
    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(f'True Positives (TP): {tp}')
    print(f'False Positives (FP): {fp}')
    print(f'True Negatives (TN): {tn}')
    print(f'False Negatives (FN): {fn}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'AUC: {auc:.4f}')
