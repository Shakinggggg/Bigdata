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
from clu_model import Autoencoder


def generate_data(path):
    data = pd.read_csv(path)
    data = calculate(data)

    # 选择特征维度
    selected_features = ['salary2', '岗位地区', '福利数量', '学历要求', '工作经验要求', '需求技能数量', '融资情况',
                         '公司人数']

    # 提取选定的特征列
    X = data[selected_features]

    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


def train(data):
    # 初始化模型和优化器
    model = Autoencoder(input_dim, hidden_dim, out_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i in range(0, data.shape[0], batch_size):
            inputs = torch.tensor(data[i:i + batch_size], dtype=torch.float32)
            optimizer.zero_grad()
            encoded, decoded = model(inputs)
            loss = criterion(decoded, inputs)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    with torch.no_grad():
        inputs = torch.tensor(data, dtype=torch.float32)
        encoded, _ = model(inputs)
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(encoded)

    # 将聚类结果添加到原始数据中
    ori_data = pd.read_csv('dealed_data3.csv')
    ori_data = calculate(ori_data)
    ori_data['Cluster'] = kmeans.labels_
    ori_data.to_csv('clustering.csv')
    # 可视化聚类结果 (如果需要)
    plt.scatter(encoded[:, 0], encoded[:, 1], c=ori_data['Cluster'], cmap='viridis', edgecolor='k', s=50)
    plt.xlabel('Latent Variable 1')
    plt.ylabel('Latent Variable 2')
    plt.title('Clustering Results')
    plt.colorbar(label='Cluster')
    plt.show()
    torch.save(model.state_dict(), 'autoencoder.pth')


def clustering(model_path):

    model = Autoencoder(input_dim, hidden_dim, out_dim)
    model.load_state_dict(torch.load(model_path))


if __name__ == '__main__':
    nor_data = generate_data('dealed_data3.csv')
    # 输入层、隐藏层、输出层、学习率、批次大小和训练轮次
    input_dim, hidden_dim, out_dim, learning_rate, batch_size, num_epochs = nor_data.shape[1], 64, 2, 0.0008, 64, 100

    # train(nor_data)

    clustering('autoencoder.pth')
