import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Rectangle
from mpl_toolkits.mplot3d import art3d

def draw_study_logic():
    fig = plt.figure(figsize=(16, 8), dpi=100)
    
    # 1. 创建左侧 3D 效果的海温场堆叠 (Correlation Operator 区域)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    
    def draw_layer(z_offset, color, label):
        # 绘制平面网格
        x, y = np.meshgrid(np.linspace(0, 10, 11), np.linspace(0, 10, 11))
        z = np.full_like(x, z_offset)
        ax1.plot_wireframe(x, y, z, color=color, alpha=0.3, linewidth=0.5)
        # 绘制模拟热力场
        ax1.plot_surface(x, y, z, color=color, alpha=0.2)
        ax1.text(11, 5, z_offset, label, fontsize=12, fontweight='bold')

    # 画两层海温场 (代表两个时间点或两个变量)
    draw_layer(z_offset=0, color='royalblue', label='SST Field $X_i$')
    draw_layer(z_offset=5, color='cornflowerblue', label='SST Field $X_j$')

    # 在两层之间画箭头和公式
    ax1.quiver(5, 5, 4.5, 0, 0, -3.5, color='darkred', arrow_length_ratio=0.1, linewidth=2)
    ax1.text(5.5, 5, 2.5, r"$\rho_{ij} = \mathrm{Pearson}(X_i, X_j)$", color='black', fontsize=14, bbox=dict(facecolor='white', alpha=0.8))
    
    # 标注过滤器
    ax1.text(5.5, 5, 1.0, r"$\mathrm{if} \ \rho > c \ \rightarrow \mathrm{Edge}$", color='red', fontsize=12, fontweight='bold')

    ax1.set_axis_off()
    ax1.set_title("Correlation Operator & Sparsification", pad=20, fontsize=15)
    ax1.view_init(elev=20, azim=-35)

    # 2. 创建右侧 稀疏邻接矩阵 (Sparse Adjacency Matrix 区域)
    ax2 = fig.add_subplot(1, 2, 2)
    N = 50
    matrix = np.zeros((N, N))
    
    # 模拟对角线附近的局部连接
    for i in range(N):
        for j in range(max(0, i-2), min(N, i+3)):
            matrix[i, j] = 0.6  # 蓝色表示局部
            
    # 模拟远离对角线的远程连通点
    remote_points = [(10, 40), (42, 8), (15, 35), (38, 12)]
    for r, c in remote_points:
        matrix[r-1:r+1, c-1:c+1] = 1.0  # 红色表示远程

    # 绘图
    cmap = plt.cm.get_cmap('Blues')
    ax2.imshow(matrix, cmap='Blues', interpolation='nearest')
    
    # 用红色圆圈圈出远程连通性
    for r, c in remote_points:
        circle = plt.Circle((c, r), 2, color='red', fill=False, linewidth=2)
        ax2.add_patch(circle)

    # 标注说明
    ax2.annotate('Local Proximity\n(Heat Diffusion)', xy=(5, 5), xytext=(15, -5),
                 arrowprops=dict(arrowstyle='->', color='blue'), color='blue')
    ax2.annotate('Remote Teleconnection\n(e.g. ENSO)', xy=(40, 10), xytext=(35, 25),
                 arrowprops=dict(arrowstyle='->', color='red'), color='red')

    ax2.set_title(r"Sparse Adjacency Matrix $A_{prior}$", fontsize=15)
    ax2.set_xlabel("Node Index $i$")
    ax2.set_ylabel("Node Index $j$")
    
    # 底部说明文字
    fig.text(0.5, 0.05, "Capturing Global Signals beyond Local Receptive Fields of ConvLSTM", 
             ha="center", fontsize=14, fontweight='bold', bbox=dict(facecolor='yellow', alpha=0.2))

    plt.tight_layout()
    plt.show()

draw_study_logic()