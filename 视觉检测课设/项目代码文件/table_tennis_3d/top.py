"""
伪俯视图生成（直接使用两个相机二维轨迹的 x 像素作为 X, Y）
=============================================================
原理：
  - Camera A 的 x 像素 → 俯视图 X
  - Camera B 的 x 像素 → 俯视图 Y
  - 时间对齐：按各自的时间戳取对应点
输出: outputs/pseudo_topview.png （轨迹曲线 + 起终点标记）
"""
import numpy as np
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 导入 trajectory.py 中的检测函数
from trajectory import get_trajectory_data

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base, 'outputs')
    os.makedirs(out_dir, exist_ok=True)

    video_dir = os.path.join(base, 'data', 'videos')
    print("正在提取二维轨迹...")
    data_A = get_trajectory_data(os.path.join(video_dir, 'A.mp4'))
    data_B = get_trajectory_data(os.path.join(video_dir, 'B.mp4'))

    if data_A is None or data_B is None:
        print("轨迹提取失败，请检查视频或检测参数。")
        return

    # 从字典中取出像素坐标和时间
    pts_A = data_A['pts']       # (N,2) 数组，第一列是 x，第二列是 y
    times_A = np.array(data_A['times'])
    pts_B = data_B['pts']
    times_B = np.array(data_B['times'])

    # 我们仅使用两个相机的 x 坐标（第一列）
    x_vals_A = pts_A[:, 0]
    x_vals_B = pts_B[:, 0]

    # 由于两相机帧率/起始时间可能不同，需要进行时间对齐
    # 方法：将各自的时间统一到一个公共时间轴（例如以秒为单位），
    # 然后对 B 的 x 值进行线性插值，使其与 A 的时间戳对齐
    if len(times_A) < 2 or len(times_B) < 2:
        print("点数不足")
        return

    # 创建公共时间轴：从两个序列的最小时间到最大时间，步长取平均帧间隔
    t_min = max(times_A[0], times_B[0])
    t_max = min(times_A[-1], times_B[-1])
    if t_min >= t_max:
        print("两个相机的时间范围无交集，无法对齐")
        return

    dt = min(np.mean(np.diff(times_A)), np.mean(np.diff(times_B)))
    t_common = np.arange(t_min, t_max, dt)

    # 对 A 和 B 的 x 序列分别在公共时间轴上插值
    x_A_interp = np.interp(t_common, times_A, x_vals_A)
    x_B_interp = np.interp(t_common, times_B, x_vals_B)

    # 现在 x_A_interp 作为俯视 X，x_B_interp 作为俯视 Y
    X = x_A_interp
    Y = 1200-x_B_interp

    # 起点：直接使用第一个有效点
    start_x = X[0]
    start_y = Y[0]

    # 绘制伪俯视图
    fig, ax = plt.subplots(figsize=(8, 8))
    # 轨迹线（用时间着色）
    points = np.array([X, Y]).T.reshape(-1, 1, 2)
    # 分段绘制以显示颜色渐变
    for i in range(len(X) - 1):
        ax.plot(X[i:i+2], Y[i:i+2], color=plt.cm.viridis(i / len(X)),
                linewidth=2, alpha=0.8)
    # 起点和终点
    ax.scatter(start_x, start_y, c='lime', s=150, marker='o',
               edgecolors='darkgreen', linewidths=2, label='Start')
    ax.scatter(X[-1], Y[-1], c='red', s=150, marker='X', linewidths=3, label='End')

    ax.set_xlabel('Pseudo X (Camera A x pixel)')
    ax.set_ylabel('Pseudo Y (Camera B x pixel)')
    ax.set_title('Pseudo Top View (A.x → X, B.x → Y)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_aspect('equal')  # 保持像素比例
    ax.grid(alpha=0.3)
    ax.invert_yaxis()  # 图像坐标系通常 y 向下，翻转后符合俯视习惯

    plt.tight_layout()
    save_path = os.path.join(out_dir, 'pseudo_topview.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"伪俯视图已保存: {save_path}")

if __name__ == '__main__':
    main()