"""
纯像素坐标系落点预测 + 伪三维轨迹展示
=======================================================================
功能：利用两台相机提取的二维像素轨迹，在图像平面上直接预测乒乓球的
落点像素位置，并构建一个伪三维图（A.x → X, B.x → Y, B.y → Z），
用于可视化球的运动过程。

核心流程：
  1. 从 trajectory.py 中获取已提取好的二维轨迹（像素坐标＋时间戳）
  2. 用二次多项式外推每个相机中球的最后像素落点
  3. 判断落点是否在预先标定的球台角点四边形内
  4. 绘制每个相机的落点示意图
  5. 通过时间戳匹配，将两个相机数据融合，生成伪三维轨迹图
  6. 输出预测报告

用法：
  1. 在代码开头填写 TABLE_PIX_A / TABLE_PIX_B（球台四个角点像素坐标）
  2. 确保 trajectory.py 在同一目录，且已经提取好二维轨迹
  3. python predict_pixel_landing.py
输出：
  outputs/pixel_landing_A.png, pixel_landing_B.png,
  outputs/pseudo_3d_trajectory.png, pixel_report.txt
"""

import numpy as np
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
import matplotlib
matplotlib.use('Agg')           # 非交互式后端，直接保存图片
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # 启用3D投影（必须导入才能使用 projection='3d'）
from trajectory import get_trajectory_data   # 复用 trajectory.py 的轨迹提取函数

# ============================================================
# 手动标定的球台角点像素坐标（顺序：左下，右下，右上，左上）
# 说明：相机视角下球台的四个顶点在图像中的像素坐标。
#       需要根据实际视频画面进行标注，并按固定顺序填写。
# ============================================================
TABLE_PIX_A = [
    (20, 695),        # 左下角
    (1842, 764),      # 右下角
    (1529, 679),      # 右上角
    (379, 643)        # 左上角
]   # 相机 A 的球台角点

TABLE_PIX_B = [
    (0, 950),         # 左下角
    (1920, 926),      # 右下角
    (1232, 672),      # 右上角
    (563, 670)        # 左上角
]   # 相机 B 的球台角点

# ============================================================
# 工具函数
# ============================================================

def is_point_in_polygon(pt, polygon):
    """
    使用射线法判断一个点是否在凸或凹多边形内部。
    参数：
        pt: (x, y) 待检测点
        polygon: 多边形顶点列表，顺序为顺时针或逆时针
    返回：
        True 表示点在多边形内部，False 表示外部
    """
    x, y = pt
    n = len(polygon)
    inside = False
    px, py = polygon[0]                     # 起始顶点
    for i in range(1, n):
        cx, cy = polygon[i]                 # 当前顶点
        # 射线法核心：判断边是否跨越点所在的水平线，且点在边的左侧
        if ((cy > y) != (py > y)) and (x < (px - cx) * (y - cy) / (py - cy) + cx):
            inside = not inside
        px, py = cx, cy
    return inside

def extrapolate_pixel(times_arr, px_arr, py_arr):
    """
    使用二次多项式对最后60%的像素轨迹进行拟合，并外推0.3秒，
    得到预测落点像素坐标。
    参数：
        times_arr: 时间戳序列（单位：秒）
        px_arr, py_arr: 对应的 x 和 y 像素坐标序列
    返回：
        (x_pred, y_pred) 预测落点像素坐标
        如果数据太少或拟合失败，返回 None
    """
    if len(times_arr) < 5:
        return None

    # 转换为 numpy 数组以便计算
    times_arr = np.array(times_arr)
    px_arr = np.array(px_arr)
    py_arr = np.array(py_arr)

    # 取后60%的点进行拟合（假设球速变化不大，末端更能反映落点趋势）
    n_fit = max(5, int(len(times_arr) * 0.6))
    t_fit = times_arr[-n_fit:] - times_arr[-n_fit]   # 时间归零，减少数值误差
    x_fit = px_arr[-n_fit:]
    y_fit = py_arr[-n_fit:]

    try:
        # 分别对 x 和 y 做二次多项式拟合
        px = np.polyfit(t_fit, x_fit, 2)
        py = np.polyfit(t_fit, y_fit, 2)
    except Exception:
        return None

    # 外推到最后一个时间点再加 0.3 秒（经验值，假设球约0.3秒内接触桌面）
    t_end = times_arr[-1] + 0.3
    x_pred = np.polyval(px, t_end - times_arr[-n_fit])
    y_pred = np.polyval(py, t_end - times_arr[-n_fit])
    return (x_pred, y_pred)

# ============================================================
# 主程序
# ============================================================
def main():
    # ---------- 路径初始化 ----------
    base = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base, 'outputs')
    os.makedirs(out_dir, exist_ok=True)

    video_dir = os.path.join(base, 'data', 'videos')

    # ---------- 1. 提取二维轨迹 ----------
    print("提取二维轨迹...")
    # 调用 trajectory.py 中的 get_trajectory_data 函数，
    # 该函数返回一个字典，包含 'pts' (Nx2 数组) 和 'times' (列表)
    data_A = get_trajectory_data(os.path.join(video_dir, 'A.mp4'))
    data_B = get_trajectory_data(os.path.join(video_dir, 'B.mp4'))

    if data_A is None or data_B is None:
        print("轨迹提取失败！请检查视频路径或检测参数。")
        return

    pts_A = data_A['pts']              # 相机 A 的 (x, y) 像素坐标
    times_A = np.array(data_A['times'])   # 时间戳（秒）
    pts_B = data_B['pts']              # 相机 B 的 (x, y) 像素坐标
    times_B = np.array(data_B['times'])

    print(f"Camera A: {len(pts_A)} 个点, Camera B: {len(pts_B)} 个点")

    # ---------- 2. 预测落点（纯像素坐标系） ----------
    pred_A = extrapolate_pixel(times_A, pts_A[:, 0], pts_A[:, 1])
    pred_B = extrapolate_pixel(times_B, pts_B[:, 0], pts_B[:, 1])

    # 判断预测的落点是否在预先标定的球台多边形内
    in_A = is_point_in_polygon(pred_A, TABLE_PIX_A) if pred_A else False
    in_B = is_point_in_polygon(pred_B, TABLE_PIX_B) if pred_B else False
    # 综合判断：两台相机均预测在球台内才认为在界内
    in_bounds = in_A and in_B

    print(f"Camera A 预测落点: {pred_A}  界内: {in_A}")
    print(f"Camera B 预测落点: {pred_B}  界内: {in_B}")
    print(f"综合判断: {'球台内' if in_bounds else '出界'}")

    # ---------- 3. 绘制每个相机的落点预测图 ----------
    for label, pts, times, pred, in_flag, corners, w, h in [
        ('A', pts_A, times_A, pred_A, in_A, TABLE_PIX_A, data_A['w'], data_A['h']),
        ('B', pts_B, times_B, pred_B, in_B, TABLE_PIX_B, data_B['w'], data_B['h'])
    ]:
        fig, ax = plt.subplots(figsize=(9, 7))

        # 绘制球台多边形（半透明绿色填充）
        poly = np.array(corners)
        ax.fill(poly[:, 0], poly[:, 1], alpha=0.1, edgecolor='green', linewidth=2)

        # 绘制所有检测到的轨迹点（用时间着色）
        if len(pts) > 0:
            ax.scatter(pts[:, 0], pts[:, 1], c=times, cmap='viridis', s=12, alpha=0.8)
            # 起点和终点用特殊标记
            ax.scatter(*pts[0], c='lime', s=120, marker='o', edgecolors='darkgreen', linewidths=2, label='Start')
            ax.scatter(*pts[-1], c='orange', s=100, marker='s', edgecolors='darkorange', linewidths=2, label='End')

        # 绘制预测落点
        if pred is not None:
            c = 'limegreen' if in_flag else 'red'
            ax.scatter(*pred, c=c, marker='X', s=250, linewidths=3, edgecolors='black',
                       zorder=10, label=f"Predicted ({'IN' if in_flag else 'OUT'})")

        ax.set_title(f'Camera {label} - Pixel Landing', fontsize=13, fontweight='bold')
        ax.set_xlim(0, w); ax.set_ylim(h, 0)   # 保持图像坐标系（y 向下）
        ax.set_xlabel('X (pixels)'); ax.set_ylabel('Y (pixels)')
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'pixel_landing_{label}.png'), dpi=150)
        plt.close()
        print(f"  {out_dir}/pixel_landing_{label}.png")

    # ---------- 4. 生成伪三维轨迹图 ----------
    """
    思路：
      - Camera A 的 x 像素  -> 三维图的 X 轴
      - Camera B 的 x 像素  -> 三维图的 Y 轴
      - Camera B 的 y 像素  -> 三维图的 Z 轴（垂直方向）
      - 时间用于着色和排序
    匹配策略：
      对 B 的每个轨迹点，在 A 中寻找时间最接近的点，
      若时间差小于阈值 match_threshold (0.1秒)，则认为是一对匹配点，
      构建一个三维点。未匹配的点直接丢弃，保证数据同步。
    """
    match_threshold = 0.1   # 秒，可调节（25fps 时为 0.04，为鲁棒性这里加大）
    X3d, Y3d, Z3d, t3d = [], [], [], []    # 分别存储匹配后的三维点坐标和对应时间

    for (t_b, (x_b, y_b)) in zip(times_B, pts_B):
        # 在 A 中查找时间最接近的点
        idx_a = np.argmin(np.abs(times_A - t_b))
        dt = abs(times_A[idx_a] - t_b)
        if dt < match_threshold:
            X3d.append(pts_A[idx_a, 0])      # A 的 x → X 轴
            Y3d.append(x_b)                  # B 的 x → Y 轴
            Z3d.append(y_b)                  # B 的 y → Z 轴
            t3d.append(t_b)                  # 保留时间用于颜色映射

    if len(X3d) == 0:
        print("没有匹配上的点，无法绘制伪三维图！")
    else:
        # 转为 numpy 数组
        X3d = np.array(X3d); Y3d = np.array(Y3d); Z3d = np.array(Z3d); t3d = np.array(t3d)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 散点图，颜色按时间
        sc = ax.scatter(X3d, Y3d, Z3d, c=t3d, cmap='viridis', s=20, alpha=0.9)

        # 标记时间最早的起点和时间最晚的终点
        start_idx = np.argmin(t3d)
        end_idx   = np.argmax(t3d)
        ax.scatter(X3d[start_idx], Y3d[start_idx], Z3d[start_idx],
                   c='lime', s=120, marker='o', edgecolors='darkgreen', linewidths=2, label='Start')
        ax.scatter(X3d[end_idx], Y3d[end_idx], Z3d[end_idx],
                   c='orange', s=120, marker='s', edgecolors='darkorange', linewidths=2, label='End')

        # 按时间顺序绘制连线，展示运动路径
        order = np.argsort(t3d)
        ax.plot(X3d[order], Y3d[order], Z3d[order], 'b-', linewidth=1, alpha=0.35)

        ax.set_xlabel('Cam A x (pixel)')
        ax.set_ylabel('Cam B x (pixel)')
        ax.set_zlabel('Cam B y (pixel)')
        ax.set_title('Pseudo-3D Trajectory (A.x, B.x, B.y)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)

        # 添加颜色条，表示时间先后
        fig.colorbar(sc, ax=ax, shrink=0.5, aspect=20, label='Time (s)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'pseudo_3d_trajectory.png'), dpi=150)
        plt.close()
        print(f"  {out_dir}/pseudo_3d_trajectory.png")

    # ---------- 5. 保存报告 ----------
    with open(os.path.join(out_dir, 'pixel_report.txt'), 'w', encoding='utf-8') as f:
        f.write("像素坐标系落点预测报告\n")
        f.write("=" * 30 + "\n")
        f.write(f"Camera A 预测落点: {pred_A}, 界内: {in_A}\n")
        f.write(f"Camera B 预测落点: {pred_B}, 界内: {in_B}\n")
        f.write(f"综合: {'球台内' if in_bounds else '出界'}\n")
        f.write(f"匹配上的伪三维点数: {len(X3d) if 'X3d' in dir() else 0}\n")
    print("报告已保存至 pixel_report.txt")

if __name__ == '__main__':
    main()