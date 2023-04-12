import matplotlib.pyplot as plt
import json
from scipy.signal import savgol_filter

generated_result_directory = "./data/generated_data/train_result/"
origin_train_result = "test.json"
my_train_result = "我的.json"

# 读取 F1 值的列表
with open(generated_result_directory + origin_train_result, 'r', encoding='utf8') as fp:
    origin_f1_list = json.load(fp)

# 读取 F1 值和 Loss 值的列表
with open(generated_result_directory + my_train_result, 'r', encoding='utf8') as fp:
    my_f1_list = json.load(fp)

# 对曲线进行平滑处理
# origin_f1_list = savgol_filter(origin_f1_list, window_length=10, polyorder=2)
# my_f1_list = savgol_filter(my_f1_list, window_length=10, polyorder=2)

# 设置图像大小
fig = plt.figure(figsize=(10, 5))

# 绘制 F1 值曲线
plt.plot(origin_f1_list, label='origin F1')
plt.plot(my_f1_list, label='my F1')

# 设置图例和标题等参数
plt.legend()
plt.title('F1')
plt.xlabel('Epoch')
plt.ylabel('F1')
plt.xticks(range(0, 50))
plt.ylim(0.5, 0.8)

# 显示图像
plt.show()
