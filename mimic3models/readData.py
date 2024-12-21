import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图风格（可选）
sns.set(style="whitegrid")

# 读取CSV文件
output_dir="E:\environment\IDE\pycharm\PycharmProjects\mimic3-benchmarks\mimic3-benchmarks\mimic3models\in_hospital_mortality"
modelName="k_lstm.n16.d0.3.dep2.bs8.ts1.0"
csv_path = os.path.join(output_dir, 'keras_logs', modelName + '.csv')
df = pd.read_csv(csv_path, sep=';')

import pandas as pd
import matplotlib.pyplot as plt

# 读取数据（假设数据已经在DataFrame df中）
# df = pd.read_csv(csv_path, sep=';')

fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制第一个Y轴的指标（例如损失）
color = 'tab:blue'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color=color)
ax1.plot(df['epoch'], df['loss'], label='Training Loss', color=color)
ax1.plot(df['epoch'], df['val_loss'], label='Validation Loss', color=color, linestyle='--')
ax1.tick_params(axis='y', labelcolor=color)

# 创建第二个Y轴
ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('Accuracy', color=color)
ax2.plot(df['epoch'], df['train_acc'], label='Training Accuracy', color=color)
ax2.plot(df['epoch'], df['val_acc'], label='Validation Accuracy', color=color, linestyle='--')
ax2.tick_params(axis='y', labelcolor=color)

# 添加标题和图例
plt.title('Training and Validation Metrics')
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)
plt.show()
# # 绘制训练和验证损失
# plt.figure(figsize=(10, 6))
# plt.plot(df['epoch'], df['loss'], label='Training Loss', marker='o')
# plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', marker='o')
#
# # 绘制准确率
# plt.plot(df['epoch'], df['train_acc'], label='Training Accuracy', marker='s')
# plt.plot(df['epoch'], df['val_acc'], label='Validation Accuracy', marker='s')
# plt.xlabel('Epoch')
# plt.ylabel('Values')
# plt.title('Training and Validation Metrics')
# plt.legend()
# plt.show()