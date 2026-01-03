#!/usr/bin/env python3
"""分析Z轴跳变问题 - 判断是轨迹规划问题还是控制器问题"""

import numpy as np
import matplotlib.pyplot as plt

def load_data(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.strip().split()
            if len(parts) >= 28:
                data.append([float(x) for x in parts])
    return np.array(data)

# 读取数据
data = load_data('control_debug.txt')

time = data[:, 0] - data[0, 0]  # 相对时间
# 期望位置
des_z = data[:, 3]
# 实际位置
act_z = data[:, 6]
# 期望速度
des_vz = data[:, 9]
# 实际速度
act_vz = data[:, 12]
# Z轴误差
err_z = data[:, 15]

# 计算期望Z位置的变化率
time_diff = np.diff(time)
time_diff[time_diff == 0] = 1e-6
des_z_rate = np.diff(des_z) / time_diff
act_z_rate = np.diff(act_z) / time_diff

print("=" * 70)
print("Z轴跳变问题分析报告")
print("=" * 70)

# 分析期望轨迹的跳变
threshold = 1.5  # m/s
jump_indices_des = np.where(np.abs(des_z_rate) > threshold)[0]
jump_indices_act = np.where(np.abs(act_z_rate) > threshold)[0]

print(f"\n【期望轨迹分析】")
print(f"  Z位置范围: {des_z.min():.3f} ~ {des_z.max():.3f} m")
print(f"  Z变化率最大值: {np.max(np.abs(des_z_rate)):.3f} m/s")
print(f"  超过阈值({threshold}m/s)的点数: {len(jump_indices_des)}")

print(f"\n【实际轨迹分析】")
print(f"  Z位置范围: {act_z.min():.3f} ~ {act_z.max():.3f} m")
print(f"  Z变化率最大值: {np.max(np.abs(act_z_rate)):.3f} m/s")
print(f"  超过阈值({threshold}m/s)的点数: {len(jump_indices_act)}")

# 判断问题来源
print("\n" + "=" * 70)
print("问题诊断")
print("=" * 70)

if np.max(np.abs(des_z_rate)) > 2.0:
    print("""
【结论】: 期望轨迹本身存在Z轴剧烈变化！

这是 **轨迹规划** 的问题，不是控制器的问题。
无论如何调整 kx 和 kv 都无法根本解决。

原因: 后端轨迹优化生成的轨迹在某些点Z轴速度过大
""")
    is_trajectory_problem = True
else:
    print("""
【结论】: 期望轨迹平滑，问题可能在控制器超调

建议增大 kv[2] 来抑制超调
""")
    is_trajectory_problem = False

# 找出跳变发生的具体位置
print("\n【跳变详情】")
# 找到误差最大的几个时刻
top_error_indices = np.argsort(np.abs(err_z))[-5:][::-1]
for idx in top_error_indices:
    print(f"  时间 {time[idx]:.2f}s: 期望Z={des_z[idx]:.3f}m, 实际Z={act_z[idx]:.3f}m, 误差={err_z[idx]:.3f}m")

# 绘图
fig, axes = plt.subplots(4, 1, figsize=(14, 12))

# 图1: Z轴位置对比
ax1 = axes[0]
ax1.plot(time, des_z, 'b-', label='Desired Z', linewidth=2)
ax1.plot(time, act_z, 'r--', label='Actual Z', linewidth=1.5)
ax1.scatter(time[top_error_indices], act_z[top_error_indices], 
            c='orange', s=100, marker='v', label='Max Error Points', zorder=5)
ax1.set_ylabel('Z Position (m)')
ax1.set_title('Z-axis Position Tracking - Identifying Jump Points')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 图2: 期望Z变化率
ax2 = axes[1]
ax2.plot(time[:-1], des_z_rate, 'b-', label='Desired dZ/dt', linewidth=1.5)
ax2.plot(time[:-1], act_z_rate, 'r-', label='Actual dZ/dt', linewidth=1, alpha=0.7)
ax2.axhline(y=threshold, color='g', linestyle='--', alpha=0.7)
ax2.axhline(y=-threshold, color='g', linestyle='--', alpha=0.7, label=f'Threshold ±{threshold}')
ax2.set_ylabel('dZ/dt (m/s)')
ax2.set_title('Z Position Rate of Change (Check if trajectory itself has jumps)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 图3: Z轴误差
ax3 = axes[2]
ax3.plot(time, err_z, 'g-', linewidth=1.5)
ax3.fill_between(time, err_z, 0, alpha=0.3, color='green')
ax3.scatter(time[top_error_indices], err_z[top_error_indices], 
            c='orange', s=100, marker='v', label='Max Error Points', zorder=5)
ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax3.set_ylabel('Z Error (m)')
ax3.set_title('Z-axis Position Error')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 图4: Z轴速度对比
ax4 = axes[3]
ax4.plot(time, des_vz, 'b-', label='Desired Vz', linewidth=2)
ax4.plot(time, act_vz, 'r--', label='Actual Vz', linewidth=1.5)
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Z Velocity (m/s)')
ax4.set_title('Z-axis Velocity Tracking')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('z_jump_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n分析图已保存: z_jump_analysis.png")

# 解决方案
print("\n" + "=" * 70)
print("解决方案")
print("=" * 70)

if is_trajectory_problem:
    print("""
【方案1】修改轨迹规划参数 (推荐)

文件: code/src/trajectory_generator/launch/demo.launch

降低规划速度和加速度:
  <param name="planning/vel" value="2.0"/>  <!-- 当前可能是3.0 -->
  <param name="planning/acc" value="1.5"/>  <!-- 当前可能是2.0 -->

【方案2】修改地图参数

降低Z轴范围或增加约束:
  <arg name="map_size_z" default="3.0"/>  <!-- 当前是4.0 -->

【方案3】控制器层面的缓解措施

虽然无法根本解决，但可以降低影响:
  - 增大 kv[2] 到 8.0~10.0 增加阻尼
  - 或者在SO3Control中添加加速度限制
""")
else:
    print("""
【方案】调整控制器参数

增大Z轴阻尼:
  kv_[2] = 8.0 ~ 10.0

或降低Z轴增益:
  kx_[2] = 6.0 ~ 8.0
""")

plt.show()
