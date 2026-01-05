import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import io

# ================= 配置区域 =================
CSV_FILENAME = 'tracking.csv'
# ===========================================

def solve_trajectory():
    # 1. 读取数据
    try:
        df = pd.read_csv(CSV_FILENAME)
        print(f"成功读取 {CSV_FILENAME}, 共 {len(df)} 行数据")
    except FileNotFoundError:
        print(f"找不到 {CSV_FILENAME}...")

    # 2. 定义常数 (根据题目描述)
    omega = 0.5        # rad/s [cite: 21]
    alpha = np.pi / 12 # rad [cite: 21]

    # 用于存储结果
    results_list = []
    
    # 3. 逐行计算
    print("开始计算坐标系转换...")
    for index, row in df.iterrows():
        t = row['t']
        
        # --- A. 获取无人机本体姿态 (World -> Body) ---
        # tracking.csv 顺序是 qx, qy, qz, qw
        q_wb_val = [row['qx'], row['qy'], row['qz'], row['qw']]
        r_wb = R.from_quat(q_wb_val)
        
        # --- B. 计算末端执行器相对姿态 (Body -> Device) ---
        # 使用题目给定的矩阵公式 (1)
        wt = omega * t
        cos_wt = np.cos(wt)
        sin_wt = np.sin(wt)
        cos_al = np.cos(alpha)
        sin_al = np.sin(alpha)
        
        # 构建旋转矩阵 R_BD
        # 注意：numpy 矩阵定义是 list of lists (行优先)
        matrix_bd = np.array([
            [cos_wt, -sin_wt * cos_al,  sin_wt * sin_al],
            [sin_wt,  cos_wt * cos_al, -cos_wt * sin_al],
            [0,       sin_al,           cos_al         ]
        ])
        
        r_bd = R.from_matrix(matrix_bd)
        
        # --- C. 链式法则求最终姿态 (World -> Device) ---
        # R_total = R_world_body * R_body_device
        r_wd = r_wb * r_bd
        
        # --- D. 提取四元数并处理评分要求 ---
        # scipy 的格式是 [x, y, z, w]
        q_wd = r_wd.as_quat()
        
        # 强制要求 qw >= 0 (题目评分点)
        if q_wd[3] < 0:
            q_wd = -q_wd
            
        # 存储结果：t, qx, qy, qz, qw
        results_list.append([t, q_wd[0], q_wd[1], q_wd[2], q_wd[3]])

    # 4. 转换为 DataFrame 方便处理
    result_df = pd.DataFrame(results_list, columns=['t', 'qx', 'qy', 'qz', 'qw'])
    
    # 5. 绘图 (题目要求: 绘制四元数的变化曲线)
    plt.figure(figsize=(10, 6))
    plt.plot(result_df['t'].values, result_df['qw'].values, label='qw (w)', linewidth=2)
    plt.plot(result_df['t'].values, result_df['qx'].values, label='qx (x)', linewidth=1.5)
    plt.plot(result_df['t'].values, result_df['qy'].values, label='qy (y)', linewidth=1.5)
    plt.plot(result_df['t'].values, result_df['qz'].values, label='qz (z)', linewidth=1.5)
    
    plt.title('End-Effector Attitude Quaternion (World Frame)', fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Quaternion Value', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图片用于报告
    plt.savefig('quaternion_plot.png', dpi=300)
    print("绘图完成，已保存为 quaternion_plot.png")
    
    # 显示前几行结果
    print("\n计算结果前 5 行预览:")
    print(result_df.head())
    
    # 如果需要保存结果 csv
    # result_df.to_csv('solution_q1.csv', index=False)

if __name__ == "__main__":
    solve_trajectory()