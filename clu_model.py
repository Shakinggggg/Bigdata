import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from calculate import calculate
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
# 中文乱码解决方法
plt.rcParams['font.family'] = ['Arial Unicode MS','Microsoft YaHei','SimHei','sans-serif']
plt.rcParams['axes.unicode_minus'] = False
font_path = "C:/Windows/Fonts/msyh.ttc"
# 读取数据
data = pd.read_csv('dealed_data3.csv')  # 替换为您的数据文件路径
data = calculate(data)
# 选择特征维度
selected_features = ['salary2', '岗位地区', '福利数量', '学历要求', '工作经验要求', '需求技能数量', '融资情况', '公司人数']

# 提取选定的特征列
X = data[selected_features]
# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#
# # 使用K均值聚类
# kmeans = KMeans(n_clusters=5, random_state=42)  # 替换为您希望的簇数
# kmeans.fit(X_scaled)
#
# # 将聚类结果添加到原始数据中
# data['Cluster'] = kmeans.labels_
# print(data['Cluster'])
# # 可视化聚类结果
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)
#
# plt.figure(figsize=(8, 6))
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data['Cluster'], cmap='viridis', edgecolor='k', s=50)
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.title('Clustering Results')
# plt.colorbar(label='Cluster')
# plt.show()
#
# data.to_csv('clustering.csv')



input_dim = X_scaled.shape[1]
hidden_dim = 64
latent_dim = 2  # 假设潜在空间维度为2
num_epochs = 100
batch_size = 32
learning_rate = 0.001


# 定义自编码器模型
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# 初始化模型和优化器
model = Autoencoder(input_dim, hidden_dim, latent_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    for i in range(0, X_scaled.shape[0], batch_size):
        inputs = torch.tensor(X_scaled[i:i+batch_size], dtype=torch.float32)
        optimizer.zero_grad()
        encoded, decoded = model(inputs)
        loss = criterion(decoded, inputs)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 提取潜在空间表示
with torch.no_grad():
    inputs = torch.tensor(X_scaled, dtype=torch.float32)
    encoded, _ = model(inputs)

# 聚类潜在空间表示 (例如，K均值聚类)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(encoded)

# 将聚类结果添加到原始数据中
data['Cluster'] = kmeans.labels_
data['X'] = encoded[:, 0]
data['Y'] = encoded[:, 1]
# 可视化聚类结果 (如果需要)
import matplotlib.pyplot as plt
plt.scatter(encoded[:, 0], encoded[:, 1], c=data['Cluster'], cmap='viridis', edgecolor='k', s=50)
plt.xlabel('Latent Variable 1')
plt.ylabel('Latent Variable 2')
plt.title('Clustering Results')
plt.colorbar(label='Cluster')
plt.show()

print(encoded)

# data.to_csv('clustering.csv')
