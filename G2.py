import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 解决 Matplotlib 中文显示问题（根据系统环境选择字体）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False     # 正常显示负号

def draw_sst_definition_cn():
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 生成模拟的全球海温场网格数据
    def create_sst_data():
        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 10, 50)
        X, Y = np.meshgrid(x, y)
        # 模拟赤道热、两极冷的分布，加入随机噪声
        Z = 20 + 10 * np.sin(Y/3) + np.random.randn(*X.shape) * 0.5
        return X, Y, Z

    # 设置时间切片的高度 (Z轴)
    # 输入步长 Tin (3层) 和 输出步长 Tout (1层)
    input_offsets = [0, 2, 4]
    output_offset = 10
    
    X, Y, Z_base = create_sst_data()

    # 1. 绘制输入的历史切片 (输入 Tin 步) 
    for i, z_pos in enumerate(input_offsets):
        # 使用 RdYlBu_r 颜色映射（红暖蓝冷）
        ax.contourf(X, Y, Z_base + i*0.5, zdir='z', offset=z_pos, cmap='RdYlBu_r', alpha=0.8)
        ax.text(11, 5, z_pos, f"历史时刻 T-{2-i}", color='blue', fontsize=10)

    # 2. 绘制预测的目标切片 (输出 Tout 步) 
    ax.contourf(X, Y, Z_base + 5, zdir='z', offset=output_offset, cmap='RdYlBu_r', alpha=0.8)
    ax.text(11, 5, output_offset, "预测时刻 T+future", color='red', fontweight='bold')

    # 3. 绘制连接箭头（代表模型处理过程，如 AGCRN） [cite: 26]
    ax.quiver(5, 5, 5, 0, 0, 4, color='black', arrow_length_ratio=0.3, label='时空预测模型 (AGCRN)')

    # 设置坐标轴标签与标题
    ax.set_zlim(0, 12)
    ax.set_title("问题定义：海温场（SST）多步预测任务示意图", fontsize=15)
    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    ax.set_zlabel("时间步 (T)")
    
    # 图例设置
    ax.legend(loc='upper left')
    
    # 调整视角以获得最佳 3D 效果
    ax.grid(False)
    ax.view_init(elev=25, azim=-60)

    plt.show()

if __name__ == "__main__":
    draw_sst_definition_cn()