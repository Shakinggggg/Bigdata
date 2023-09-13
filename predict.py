from calculate import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 使用DejaVu Sans或其他可用字体


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_size2):
        super(MultiLayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


def generate_data(ori_data):
    selected_column = ['岗位地区', '福利数量', '学历要求', '工作经验要求', '需求技能数量', '融资情况', '公司人数']
    input_x = ori_data[selected_column].values
    input_y = ori_data[['salary2']].values
    return input_x, input_y


def train(x, y, num_epoch, learn_rate, batch_size, hidden_size, output_size, hidden_size2):
    input_size = x.shape[1]
    model = MultiLayerPerceptron(input_size, hidden_size, output_size, hidden_size2)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learn_rate)

    losses = []
    for epoch in range(num_epoch):
        total_loss = 0
        for i in range(0, X.shape[0], batch_size):
            batch_X = torch.FloatTensor(X[i:i + batch_size])
            batch_y = torch.FloatTensor(y[i:i + batch_size])

            # 前向传播
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 记录每个epoch的损失
        losses.append(total_loss / (X.shape[0] // batch_size))
        print(f'Epoch [{epoch + 1}/{num_epoch}], Loss: {total_loss / (X.shape[0] // batch_size)}')

    # 打印最终模型的权重和偏置
    # print("Final model's state_dict:")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor])

    torch.save(model.state_dict(), 'model_params.pth')
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()


def normalize(pre_data):
    ori_data = load_data('dealed_data3.csv')
    max_list, min_list = [0 for _ in range(7)], [0 for _ in range(7)]

    loc_dict = ori_data['岗位地区'].value_counts().to_dict()
    max_list[0], min_list[0] = max(list(ori_data['岗位地区'].value_counts())), min(list(ori_data['岗位地区'].value_counts()))
    pre_data[0] = (loc_dict[pre_data[0]] - min_list[0]) / (max_list[0] - min_list[0])

    max_list[1], min_list[1] = max(list(ori_data['福利数量'])), min(list(ori_data['福利数量']))
    pre_data[1] = (pre_data[1] - min_list[1]) / (max_list[1] - min_list[1])

    edu_dict = {'学历不限': 8, '初中及以下': 7, '中专/中技': 6, '高中': 5, '大专': 4, '本科': 3, '硕士': 2, '博士': 1}
    max_list[2], min_list[2] = 8, 1
    pre_data[2] = (edu_dict[pre_data[2]] - min_list[2]) / (max_list[2] - min_list[2])

    max_list[3], min_list[3] = 10, 4
    exp_dict = {'经验不限': 10, '在校/应届': 9, '应届生': 9, '1年以内': 8, '1-3年': 7, '3-5年': 6, '5-10年': 5,
                '10年以上': 4}
    pre_data[3] = (exp_dict[pre_data[3]] - min_list[3]) / (max_list[3] - min_list[3])

    max_list[4], min_list[4] = max(list(ori_data['需求技能数量'])), min(list(ori_data['需求技能数量']))
    pre_data[4] = (pre_data[4] - min_list[4]) / (max_list[4] - min_list[4])

    rongzi_dict = {'已上市': 10, 'D轮及以上': 9, 'C轮': 8, 'B轮': 7, 'A轮': 6, '天使轮': 5, '未融资': 4, '未知': 4,
                   '不需要融资': 4}
    max_list[5], min_list[5] = 10, 4
    pre_data[5] = (rongzi_dict[pre_data[5]] - min_list[5]) / (max_list[5] - min_list[5])

    size_dict = {'10000人以上': 10, '1000-9999人': 9, '500-999人': 8, '100-499人': 7, '20-99人': 6, '0-20人': 5,
                 '未知': 4}
    max_list[6], min_list[6] = 10, 4
    pre_data[6] = (size_dict[pre_data[6]] - min_list[6]) / (max_list[6] - min_list[6])

    return pre_data, max(list(ori_data['salary2'])), min(list(ori_data['salary2']))


def predict(input_size, output_size, hidden_size, hidden_size2, input_x):
    predict_model = MultiLayerPerceptron(input_size, hidden_size, output_size, hidden_size2)
    predict_model.load_state_dict(torch.load('model_params.pth'))
    predict_model.eval()

    data = input_x
    data, max_sal, min_sal = normalize(data)
    data_tensor = torch.FloatTensor(data)
    prediction = predict_model(data_tensor)
    prediction = prediction * (max_sal - min_sal) + min_sal
    return float(prediction)


if __name__ == '__main__':
    data = load_data('dealed_data3.csv')
    numerical_data = calculate(data)
    X, Y = generate_data(numerical_data)
    epoch, learn_rate, batch_size, hidden_size, out_size, hidden_size2 = 100, 0.0008, 32, 64, 1, 32
    # train(X, Y, epoch, learn_rate, batch_size, hidden_size, out_size, hidden_size2)
    test_X = ['上海', 5, '本科', '经验不限', 3, '不需要融资', '20-99人']
    print(predict(X.shape[1], out_size, hidden_size, hidden_size2, test_X))

