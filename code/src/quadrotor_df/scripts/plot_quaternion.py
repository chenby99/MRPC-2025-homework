#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt
import pandas as pd

# 读取CSV文件
csv_path = '/home/bonnie/File/noetic_ws/MRPC-2025-homework/solutions/df_quaternion.csv'
data = pd.read_csv(csv_path)

# 提取数据
t = data['t'].values
qx = data['x'].values
qy = data['y'].values
qz = data['z'].values
qw = data['w'].values

# 创建图形：修改为 4行1列 的纵向堆叠，共享x轴
fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
# 也可以统一设置整个图的标题
# fig.suptitle('End-Effector Attitude Quaternion Components', fontsize=16, fontweight='bold')

# 定义通用的网格样式，与参考代码一致
grid_style = {'linestyle': '--', 'alpha': 0.7}

# 绘制qx (参考代码中第二个绘制，对应颜色为橙色/C1)
axes[0].plot(t, qx, color='tab:orange', linewidth=1.5, label='qx (x)')
axes[0].set_ylabel('qx', fontsize=12)
axes[0].set_title('Quaternion X Component', fontsize=13)
axes[0].grid(True, **grid_style)
axes[0].legend(loc='upper right')

# 绘制qy (参考代码中第三个绘制，对应颜色为绿色/C2)
axes[1].plot(t, qy, color='tab:green', linewidth=1.5, label='qy (y)')
axes[1].set_ylabel('qy', fontsize=12)
axes[1].set_title('Quaternion Y Component', fontsize=13)
axes[1].grid(True, **grid_style)
axes[1].legend(loc='upper right')

# 绘制qz (参考代码中第四个绘制，对应颜色为红色/C3)
axes[2].plot(t, qz, color='tab:red', linewidth=1.5, label='qz (z)')
axes[2].set_ylabel('qz', fontsize=12)
axes[2].set_title('Quaternion Z Component', fontsize=13)
axes[2].grid(True, **grid_style)
axes[2].legend(loc='upper right')

# 绘制qw (参考代码中第一个绘制，对应颜色为蓝色/C0，线宽为2)
axes[3].plot(t, qw, color='tab:blue', linewidth=2, label='qw (w)')
axes[3].set_ylabel('qw', fontsize=12)
axes[3].set_title('Quaternion W Component', fontsize=13)
axes[3].grid(True, **grid_style)
axes[3].legend(loc='upper right')

# 只在最后一个子图设置X轴标签
axes[3].set_xlabel('Time (s)', fontsize=12)

plt.tight_layout()

# 保存分离的曲线图
output_path = '/home/bonnie/File/noetic_ws/MRPC-2025-homework/solutions/quaternion_plot.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"四元数变化图(纵向堆叠)已保存到: {output_path}")

# --- 下面是绘制在一张图上的代码 (保持原来的逻辑，但可选更新颜色以匹配) ---

fig2, ax = plt.subplots(figsize=(10, 6))
# 使用与上面一致的颜色和样式
ax.plot(t, qw, color='tab:blue', linewidth=2, label='qw (w)')
ax.plot(t, qx, color='tab:orange', linewidth=1.5, label='qx (x)')
ax.plot(t, qy, color='tab:green', linewidth=1.5, label='qy (y)')
ax.plot(t, qz, color='tab:red', linewidth=1.5, label='qz (z)')

ax.set_title('End-Effector Attitude Quaternion (World Frame)', fontsize=14)
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Quaternion Value', fontsize=12)
ax.legend()
ax.grid(True, **grid_style)
plt.tight_layout()

# 保存组合图
output_path2 = '/home/bonnie/File/noetic_ws/MRPC-2025-homework/solutions/quaternion_combined_plot.png'
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f"四元数组合图已保存到: {output_path2}")

# 验证四元数归一化 (保持不变)
norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
print(f"\n四元数归一化检查:")
print(f"最小模: {np.min(norm):.10f}")
print(f"最大模: {np.max(norm):.10f}")
print(f"平均模: {np.mean(norm):.10f}")
print(f"所有四元数均已归一化: {np.allclose(norm, 1.0)}")

# 检查qw >= 0 (保持不变)
print(f"\nqw >= 0 检查:")
print(f"最小 qw: {np.min(qw):.10f}")
print(f"所有 qw >= 0: {np.all(qw >= -1e-10)}")