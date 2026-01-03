#!/usr/bin/env python3
"""
控制器调参可视化工具
用于分析 control_debug.txt 中的数据，辅助 PD 控制器调参
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 需要导入3D支持
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_data(filepath):
    """加载调试数据"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.strip().split()
            if len(parts) >= 28:
                data.append([float(x) for x in parts])
    return np.array(data)

def plot_position_tracking(data, save_path=None):
    """绘制位置跟踪图"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    time = data[:, 0] - data[0, 0]  # 相对时间
    labels = ['X', 'Y', 'Z']
    colors_des = ['b', 'g', 'r']
    colors_act = ['c', 'lime', 'orange']
    
    for i, (ax, label) in enumerate(zip(axes, labels)):
        des_col = 1 + i  # des_x, des_y, des_z
        pos_col = 4 + i  # pos_x, pos_y, pos_z
        
        ax.plot(time, data[:, des_col], colors_des[i], label=f'Desired {label}', linewidth=1.5)
        ax.plot(time, data[:, pos_col], colors_act[i], label=f'Actual {label}', linewidth=1.5, linestyle='--')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'{label} Position (m)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{label}-axis Position Tracking')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Position tracking plot saved to: {save_path}")
    plt.show()

def plot_position_error(data, save_path=None):
    """绘制位置误差图"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    time = data[:, 0] - data[0, 0]
    labels = ['X', 'Y', 'Z']
    colors = ['b', 'g', 'r']
    
    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        err_col = 13 + i  # err_x, err_y, err_z
        
        ax.plot(time, data[:, err_col], color, linewidth=1)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.fill_between(time, data[:, err_col], 0, alpha=0.3, color=color)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'{label} Error (m)')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{label}-axis Position Error')
        
        # 显示统计信息
        rmse = np.sqrt(np.mean(data[:, err_col]**2))
        max_err = np.max(np.abs(data[:, err_col]))
        ax.text(0.02, 0.95, f'RMSE: {rmse:.4f} m\nMax: {max_err:.4f} m', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Position error plot saved to: {save_path}")
    plt.show()

def plot_velocity_tracking(data, save_path=None):
    """绘制速度跟踪图"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    time = data[:, 0] - data[0, 0]
    labels = ['X', 'Y', 'Z']
    
    for i, (ax, label) in enumerate(zip(axes, labels)):
        des_col = 7 + i   # des_vx, des_vy, des_vz
        vel_col = 10 + i  # vel_x, vel_y, vel_z
        
        ax.plot(time, data[:, des_col], 'b', label=f'Desired V{label}', linewidth=1.5)
        ax.plot(time, data[:, vel_col], 'r', label=f'Actual V{label}', linewidth=1.5, linestyle='--')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'V{label} (m/s)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{label}-axis Velocity Tracking')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Velocity tracking plot saved to: {save_path}")
    plt.show()

def plot_velocity_error(data, save_path=None):
    """绘制速度误差图"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    time = data[:, 0] - data[0, 0]
    labels = ['X', 'Y', 'Z']
    colors = ['b', 'g', 'r']
    
    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        verr_col = 16 + i  # verr_x, verr_y, verr_z
        
        ax.plot(time, data[:, verr_col], color, linewidth=1)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.fill_between(time, data[:, verr_col], 0, alpha=0.3, color=color)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'V{label} Error (m/s)')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{label}-axis Velocity Error')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Velocity error plot saved to: {save_path}")
    plt.show()

def plot_control_force(data, save_path=None):
    """绘制控制力图"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    time = data[:, 0] - data[0, 0]
    labels = ['X', 'Y', 'Z']
    colors = ['b', 'g', 'r']
    
    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        force_col = 19 + i  # force_x, force_y, force_z
        
        ax.plot(time, data[:, force_col], color, linewidth=1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Force {label} (N)')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{label}-axis Control Force')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Control force plot saved to: {save_path}")
    plt.show()

def plot_3d_trajectory(data, save_path=None):
    """绘制3D轨迹图"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 期望轨迹
    ax.plot(data[:, 1], data[:, 2], data[:, 3], 'b-', label='Desired', linewidth=2)
    # 实际轨迹
    ax.plot(data[:, 4], data[:, 5], data[:, 6], 'r--', label='Actual', linewidth=2)
    
    # 标记起点和终点
    ax.scatter(data[0, 1], data[0, 2], data[0, 3], c='g', s=100, marker='o', label='Start')
    ax.scatter(data[-1, 1], data[-1, 2], data[-1, 3], c='m', s=100, marker='*', label='End')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()
    ax.set_title('3D Trajectory Comparison')
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"3D trajectory plot saved to: {save_path}")
    plt.show()

def plot_error_analysis(data, save_path=None):
    """综合误差分析图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    time = data[:, 0] - data[0, 0]
    
    # 1. 总位置误差
    ax1 = axes[0, 0]
    total_pos_err = np.sqrt(data[:, 13]**2 + data[:, 14]**2 + data[:, 15]**2)
    ax1.plot(time, total_pos_err, 'b', linewidth=1)
    ax1.fill_between(time, total_pos_err, 0, alpha=0.3)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Total Position Error (m)')
    ax1.set_title('Total Position Error (Euclidean)')
    ax1.grid(True, alpha=0.3)
    rmse = np.sqrt(np.mean(total_pos_err**2))
    ax1.text(0.02, 0.95, f'RMSE: {rmse:.4f} m', transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. 各轴位置误差对比
    ax2 = axes[0, 1]
    ax2.plot(time, np.abs(data[:, 13]), 'b', label='|X Error|', alpha=0.7)
    ax2.plot(time, np.abs(data[:, 14]), 'g', label='|Y Error|', alpha=0.7)
    ax2.plot(time, np.abs(data[:, 15]), 'r', label='|Z Error|', alpha=0.7)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Absolute Error (m)')
    ax2.set_title('Position Error by Axis')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 误差直方图
    ax3 = axes[1, 0]
    ax3.hist(data[:, 13], bins=50, alpha=0.5, label='X', color='b')
    ax3.hist(data[:, 14], bins=50, alpha=0.5, label='Y', color='g')
    ax3.hist(data[:, 15], bins=50, alpha=0.5, label='Z', color='r')
    ax3.set_xlabel('Position Error (m)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Error Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 相位图（位置误差 vs 速度误差）
    ax4 = axes[1, 1]
    ax4.scatter(data[:, 13], data[:, 16], alpha=0.3, s=1, c='b', label='X')
    ax4.scatter(data[:, 14], data[:, 17], alpha=0.3, s=1, c='g', label='Y')
    ax4.scatter(data[:, 15], data[:, 18], alpha=0.3, s=1, c='r', label='Z')
    ax4.set_xlabel('Position Error (m)')
    ax4.set_ylabel('Velocity Error (m/s)')
    ax4.set_title('Phase Portrait (Pos Error vs Vel Error)')
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax4.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Error analysis plot saved to: {save_path}")
    plt.show()

def print_statistics(data):
    """打印统计信息"""
    print("\n" + "="*60)
    print("控制器性能统计报告")
    print("="*60)
    
    # 位置误差
    pos_err_x = data[:, 13]
    pos_err_y = data[:, 14]
    pos_err_z = data[:, 15]
    total_pos_err = np.sqrt(pos_err_x**2 + pos_err_y**2 + pos_err_z**2)
    
    print("\n【位置误差】")
    print(f"  X轴:  RMSE = {np.sqrt(np.mean(pos_err_x**2)):.4f} m, Max = {np.max(np.abs(pos_err_x)):.4f} m")
    print(f"  Y轴:  RMSE = {np.sqrt(np.mean(pos_err_y**2)):.4f} m, Max = {np.max(np.abs(pos_err_y)):.4f} m")
    print(f"  Z轴:  RMSE = {np.sqrt(np.mean(pos_err_z**2)):.4f} m, Max = {np.max(np.abs(pos_err_z)):.4f} m")
    print(f"  总计: RMSE = {np.sqrt(np.mean(total_pos_err**2)):.4f} m, Max = {np.max(total_pos_err):.4f} m")
    
    # 速度误差
    vel_err_x = data[:, 16]
    vel_err_y = data[:, 17]
    vel_err_z = data[:, 18]
    
    print("\n【速度误差】")
    print(f"  X轴:  RMSE = {np.sqrt(np.mean(vel_err_x**2)):.4f} m/s, Max = {np.max(np.abs(vel_err_x)):.4f} m/s")
    print(f"  Y轴:  RMSE = {np.sqrt(np.mean(vel_err_y**2)):.4f} m/s, Max = {np.max(np.abs(vel_err_y)):.4f} m/s")
    print(f"  Z轴:  RMSE = {np.sqrt(np.mean(vel_err_z**2)):.4f} m/s, Max = {np.max(np.abs(vel_err_z)):.4f} m/s")
    
    # 控制增益
    print("\n【当前控制增益】")
    print(f"  kx = [{data[0, 22]:.1f}, {data[0, 23]:.1f}, {data[0, 24]:.1f}]")
    print(f"  kv = [{data[0, 25]:.1f}, {data[0, 26]:.1f}, {data[0, 27]:.1f}]")
    
    # 调参建议
    print("\n【调参建议】")
    
    # 分析振荡
    for i, axis in enumerate(['X', 'Y', 'Z']):
        err = data[:, 13 + i]
        # 计算过零点数量来判断振荡
        zero_crossings = np.sum(np.diff(np.sign(err)) != 0)
        oscillation_freq = zero_crossings / (data[-1, 0] - data[0, 0])
        
        if oscillation_freq > 2:  # 振荡频率高
            print(f"  {axis}轴: 振荡明显 (过零频率={oscillation_freq:.1f}Hz), 建议增大 kv[{i}]")
        elif np.max(np.abs(err)) > 0.5:  # 误差大
            print(f"  {axis}轴: 误差较大 (max={np.max(np.abs(err)):.2f}m), 建议增大 kx[{i}]")
        else:
            print(f"  {axis}轴: 性能良好")
    
    print("="*60)

def main():
    # 数据文件路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, 'control_debug.txt')
    
    if not os.path.exists(data_file):
        print(f"错误: 找不到数据文件 {data_file}")
        print("请先运行仿真以生成调试数据")
        return
    
    print(f"正在加载数据: {data_file}")
    data = load_data(data_file)
    
    if len(data) == 0:
        print("错误: 数据文件为空")
        return
    
    print(f"已加载 {len(data)} 条数据记录")
    
    # 打印统计信息
    print_statistics(data)
    
    # 保存图像的目录
    plot_dir = script_dir
    
    # 生成所有图像
    print("\n正在生成图像...")
    
    plot_position_tracking(data, os.path.join(plot_dir, 'pos_tracking.png'))
    plot_position_error(data, os.path.join(plot_dir, 'pos_error.png'))
    plot_velocity_tracking(data, os.path.join(plot_dir, 'vel_tracking.png'))
    plot_velocity_error(data, os.path.join(plot_dir, 'vel_error.png'))
    plot_control_force(data, os.path.join(plot_dir, 'control_force.png'))
    plot_3d_trajectory(data, os.path.join(plot_dir, '3d_trajectory.png'))
    plot_error_analysis(data, os.path.join(plot_dir, 'error_analysis.png'))
    
    print("\n所有图像已生成完成!")

if __name__ == '__main__':
    main()
