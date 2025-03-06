import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# 生成示例历史请求数据，假设有50个任务内容，每个内容有100条历史请求数据
np.random.seed(123)
historical_data = np.random.randint(0, 100, size=(100, 50))
time_series_data = pd.DataFrame(historical_data, columns=[f'Task_{i}' for i in range(1, 51)])

# print(time_series_data)
# 预测步数
forecast_steps = 10

# 存储所有任务内容的预测结果
predictions = []

# 构建并训练每个任务内容的ARIMA模型，并进行预测
for task in time_series_data.columns:
    model = ARIMA(time_series_data[task], order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_steps)
    predictions.append(forecast)

# 转换为DataFrame
predictions_df = pd.DataFrame(predictions, index=time_series_data.columns).T

# 计算预测概率（归一化处理）
total_forecasts = predictions_df.sum().sum()
probability_forecast = predictions_df.sum() / total_forecasts

# 打印预测概率
print(probability_forecast)

# 绘制预测概率
plt.figure(figsize=(12, 6))
plt.bar(probability_forecast.index, probability_forecast)
plt.xlabel('content')
plt.ylabel('probability')
plt.title('')
# plt.show()
