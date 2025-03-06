import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split


def generate_vehicle_content_data_with_peaks(num_samples, num_contents, sequence_length):
    np.random.seed(42)
    data = []

    # 定义高峰期和非高峰期的时间段
    peak_hours = [8, 9, 17, 18]  # 早晚高峰
    off_peak_hours = [12, 13, 14]  # 非高峰时段

    for _ in range(num_samples):
        sequence = []
        for t in range(sequence_length):
            # 模拟时间 (简单假设t % 24为一天中的小时)
            current_hour = t % 24

            # 高峰期请求概率更高
            if current_hour in peak_hours:
                content_requests = np.random.choice([0, 1], size=(num_contents,), p=[0.2, 0.8])
            elif current_hour in off_peak_hours:
                content_requests = np.random.choice([0, 1], size=(num_contents,), p=[0.8, 0.2])
            else:
                content_requests = np.random.choice([0, 1], size=(num_contents,), p=[0.7, 0.3])

            # 模拟某些内容更频繁地被请求
            if np.random.rand() < 0.1:  # 10% 的时间有特定的高频任务
                content_requests[0] = 1  # 例如，任务0 是一个高频率任务

            sequence.append(content_requests)

        data.append(sequence)

    data = np.array(data)
    return data

class ContentRequestLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(ContentRequestLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 获取最后一个时间步的输出
        out = self.fc(out)
        out = self.softmax(out)
        return out


def probability_forecast(content_num):
    num_samples = 1000
    sequence_length = 50
    # 生成数据
    data = generate_vehicle_content_data_with_peaks(num_samples, content_num, sequence_length)
    # 数据处理
    X = data[:, :-1, :]
    Y = data[:, -1, :]  # 预测最后一个时间步的内容请求

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # 转换为 PyTorch 张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32)

    # 模型参数
    input_size = content_num
    hidden_size = 300
    num_layers = 2
    output_size = content_num

    # 初始化模型
    model = ContentRequestLSTM(input_size, hidden_size, num_layers, output_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, Y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = model(X_test)

        # 打印预测的概率分布
        print("Sample Predictions (probabilities):")
        probabilities = predictions.numpy()[:5]
        pro = probabilities.tolist()
        return pro[random.randint(0, 4)]


