"""
三维轨迹重建与落点预测
============================================
用法: python predict.py

前提: 需先运行 calibrate.py 和 trajectory.py
利用双目标定参数重建3D轨迹，拟合物线并预测落点。

输出: outputs/3d_trajectory.png, landing_topview.png,
      outputs/trajectory_3d.csv, report.txt
"""
import cv2, numpy as np, os, sys, yaml
sys.stdout.reconfigure(encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from collections import deque

# ============================================================
# 参数
# ============================================================
TABLE_L, TABLE_W, TABLE_H = 2.74, 1.525, 0.76
G = 9.8

YELLOW_LOWER = np.array([17, 60, 60], dtype=np.uint8)
YELLOW_UPPER = np.array([40, 255, 255], dtype=np.uint8)
MIN_AREA, MAX_AREA = 15, 300
MIN_CIRCULARITY = 0.7
MIN_FILL_RATIO = 0.7
SCORE_THRESHOLD = 0.6


class BallDetector:
    def __init__(self):
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=16, detectShadows=False)
        self.prev_frames = deque(maxlen=3)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def preprocess(self, frame):
        return cv2.GaussianBlur(frame, (5, 5), 1.0)

    def detect_motion(self, frame):
        fg = cv2.morphologyEx(self.bg_sub.apply(frame), cv2.MORPH_OPEN, self.kernel)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_frames.append(gray)
        diff = np.zeros_like(fg)
        if len(self.prev_frames) >= 3:
            f2, f1, f0 = list(self.prev_frames)
            d1 = cv2.threshold(cv2.absdiff(f2, f1), 25, 255, cv2.THRESH_BINARY)[1]
            d2 = cv2.threshold(cv2.absdiff(f1, f0), 25, 255, cv2.THRESH_BINARY)[1]
            diff = cv2.bitwise_and(d1, d2)
        return cv2.dilate(cv2.bitwise_or(fg, diff), self.kernel, iterations=1)

    def detect_color(self, frame):
        return cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), YELLOW_LOWER, YELLOW_UPPER)

    def detect(self, frame):
        processed = self.preprocess(frame)
        motion = self.detect_motion(processed)
        color = self.detect_color(processed)
        candidate = cv2.bitwise_and(motion, color)
        if cv2.countNonZero(candidate) < 5:
            if cv2.countNonZero(color) < 200:
                candidate = color
            else:
                return None, 0.0
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate, 8)
        best_score, best_center = -1, None
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < MIN_AREA or area > MAX_AREA: continue
            comp = (labels == i).astype(np.uint8)
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts: continue
            cnt = max(cnts, key=cv2.contourArea)
            peri = cv2.arcLength(cnt, True)
            if peri < 1: continue
            circ = 4 * np.pi * area / (peri * peri)
            if circ < MIN_CIRCULARITY: continue
            _, radius = cv2.minEnclosingCircle(cnt)
            if radius > 0:
                fill_ratio = area / (np.pi * radius * radius)
                if fill_ratio < MIN_FILL_RATIO: continue
            area_s = 1.0 if 10 < area < 300 else max(0, 1 - abs(area-50)/200)
            if area_s <= 0: continue
            score = area_s * 0.4 + circ * 0.6
            if score > best_score:
                best_score = score
                cx, cy = centroids[i]
                x0, y0 = max(0, int(cx)-3), max(0, int(cy)-3)
                x1, y1 = min(processed.shape[1], int(cx)+4), min(processed.shape[0], int(cy)+4)
                roi = cv2.cvtColor(processed[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
                yy, xx = np.mgrid[y0:y1, x0:x1]
                if roi.sum() > 0:
                    cx = (xx * roi).sum() / roi.sum()
                    cy = (yy * roi).sum() / roi.sum()
                best_center = (float(cx), float(cy))
        if best_center and best_score > SCORE_THRESHOLD:
            return best_center, best_score
        return None, 0.0


def traingulate_3d(det_A, det_B, K_A, K_B, R, T):
    """双目三角化：按时间戳匹配 + 线性三角化"""
    P_A = K_A @ np.hstack([np.eye(3), np.zeros((3,1))])
    P_B = K_B @ np.hstack([R, T.reshape(3,1)])

    valid_A = [(ts, p) for _, ts, p, _ in det_A if p is not None]
    valid_B = [(ts, p) for _, ts, p, _ in det_B if p is not None]
    if not valid_A or not valid_B: return None

    results = []
    for ts_A, pA in valid_A:
        best_dt, best = 0.3, None
        for ts_B, pB in valid_B:
            dt = abs(ts_A - ts_B)
            if dt < best_dt:
                best_dt = dt
                best = pB
        if best is not None:
            pts = cv2.triangulatePoints(P_A, P_B,
                np.array([[pA[0]],[pA[1]]], np.float32),
                np.array([[best[0]],[best[1]]], np.float32))
            X = pts[:3,0] / pts[3,0]
            results.append((ts_A, X))

    if len(results) < 4: return None
    arr = np.array([(t, x[0], x[1], x[2]) for t, x in results])

    # Savitzky-Golay 平滑
    for i in range(1, 4):
        w = min(7, len(arr)-2); w = w if w%2==1 else w+1
        if len(arr) > w:
            arr[:,i] = savgol_filter(arr[:,i], w, min(3, w-1))
    return arr


def fit_parabolic(trajectory):
    """抛物线拟合 + 落点预测"""
    if trajectory is None or len(trajectory) < 8: return None

    t_rel = trajectory[:,0] - trajectory[0,0]
    obs = trajectory[:,1:4]

    def parabolic(t, X0, Vx0, Y0, Vy0, Z0, Vz0):
        return np.column_stack([
            X0+Vx0*t, Y0+Vy0*t, Z0+Vz0*t - 0.5*G*t**2]).ravel()

    p0 = [obs[0,0], (obs[-1,0]-obs[0,0])/t_rel[-1] if t_rel[-1]>0 else 0,
          obs[0,1], (obs[-1,1]-obs[0,1])/t_rel[-1] if t_rel[-1]>0 else 0,
          obs[0,2], (obs[-1,2]-obs[0,2])/t_rel[-1] if t_rel[-1]>0 else 0]
    try:
        popt, _ = curve_fit(parabolic, t_rel, obs.ravel(), p0=p0, maxfev=5000)
    except:
        return None

    X0, Vx0, Y0, Vy0, Z0, Vz0 = popt
    pred = parabolic(t_rel, *popt).reshape(-1,3)
    rmse = np.sqrt(np.mean((obs - pred)**2))
    ss_res = np.sum((obs - pred)**2)
    ss_tot = np.sum((obs - np.mean(obs, axis=0))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0

    # 落点: 解 Z = TABLE_H
    a, b, c = 0.5*G, -Vz0, TABLE_H - Z0
    disc = b**2 - 4*a*c
    if disc < 0: return None
    t_land = (-b + np.sqrt(disc)) / (2*a)
    if t_land <= 0: t_land = (-b - np.sqrt(disc)) / (2*a)
    if t_land <= 0: return None

    return {'X0':X0, 'Vx0':Vx0, 'Y0':Y0, 'Vy0':Vy0, 'Z0':Z0, 'Vz0':Vz0,
            'rmse':rmse, 'r2':r2, 't_land':t_land,
            'X_land': X0+Vx0*t_land, 'Y_land': Y0+Vy0*t_land,
            'in_bounds': 0 <= X0+Vx0*t_land <= TABLE_L and 0 <= Y0+Vy0*t_land <= TABLE_W}


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base, 'outputs')
    calib_file = os.path.join(base, 'calibration.yaml')
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(calib_file):
        print("错误: 请先运行 calibrate.py 生成 calibration.yaml!")
        return

    print("=" * 50)
    print("  三维轨迹重建 + 落点预测")
    print("=" * 50)

    # 加载标定
    with open(calib_file, 'r', encoding='utf-8') as f:
        calib = yaml.load(f, Loader=yaml.FullLoader)

    K_A_c = np.array(calib['K_A']); K_B_c = np.array(calib['K_B'])
    R = np.array(calib['R']); T = np.array(calib['T'])
    cw_a, ch_a = calib['image_size_A']; cw_b, ch_b = calib['image_size_B']

    # 缩放内参到视频分辨率
    video_dir = os.path.join(base, 'data', 'videos')
    sizes = {}
    for label in ['A', 'B']:
        cap = cv2.VideoCapture(os.path.join(video_dir, f'{label}.mp4'))
        sizes[label] = (int(cap.get(3)), int(cap.get(4)))
        cap.release()

    sx_a, sy_a = sizes['A'][0]/cw_a, sizes['A'][1]/ch_a
    sx_b, sy_b = sizes['B'][0]/cw_b, sizes['B'][1]/ch_b
    K_A = K_A_c.copy(); K_A[0,0]*=sx_a; K_A[1,1]*=sy_a; K_A[0,2]*=sx_a; K_A[1,2]*=sy_a
    K_B = K_B_c.copy(); K_B[0,0]*=sx_b; K_B[1,1]*=sy_b; K_B[0,2]*=sx_b; K_B[1,2]*=sy_b

    print(f"Camera A: calib {cw_a}x{ch_a} -> video {sizes['A'][0]}x{sizes['A'][1]}")
    print(f"Camera B: calib {cw_b}x{ch_b} -> video {sizes['B'][0]}x{sizes['B'][1]}")
    print(f"Baseline: {np.linalg.norm(T):.3f}m\n")

    # 检测
    print("球体检测 + 三角化...")
    all_det = {}
    for label in ['A', 'B']:
        cap = cv2.VideoCapture(os.path.join(video_dir, f'{label}.mp4'))
        fps = cap.get(5)
        det = BallDetector()
        detections = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            c, conf = det.detect(frame)
            detections.append((idx, idx/fps, c, conf))
            idx += 1
        cap.release()
        n = sum(1 for d in detections if d[2] is not None)
        print(f"  Camera {label}: {n}/{idx} detections ({100*n/idx:.1f}%)")
        all_det[label] = detections

    # 3D重建
    trajectory = traingulate_3d(all_det['A'], all_det['B'], K_A, K_B, R, T)
    if trajectory is None:
        print("错误: 三角化失败 (匹配点不足)")
        return
    print(f"  3D 轨迹点: {len(trajectory)}")

    X, Y, Z = trajectory[:,1], trajectory[:,2], trajectory[:,3]
    print(f"  X: [{X.min():.2f}, {X.max():.2f}] m")
    print(f"  Y: [{Y.min():.2f}, {Y.max():.2f}] m")
    print(f"  Z: [{Z.min():.2f}, {Z.max():.2f}] m")

    # 拟合 + 预测
    print("\n轨迹拟合...")
    fit = fit_parabolic(trajectory)

    if fit:
        print(f"  R2 = {fit['r2']:.3f}")
        print(f"  RMSE = {fit['rmse']:.3f} m")
        print(f"  初始速度: ({fit['Vx0']:.2f}, {fit['Vy0']:.2f}, {fit['Vz0']:.2f}) m/s")
        print(f"  落点: X={fit['X_land']:.3f}, Y={fit['Y_land']:.3f} m")
        print(f"  状态: {'球台内' if fit['in_bounds'] else '出界'}")
    else:
        print("  拟合失败!")

    # ============================================================
    # 图1: 3D轨迹图
    # ============================================================
    print("\n生成 3D 可视化...")
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 球台
    tv = np.array([[0,0,TABLE_H],[TABLE_L,0,TABLE_H],[TABLE_L,TABLE_W,TABLE_H],[0,TABLE_W,TABLE_H]])
    ax.plot_trisurf(tv[:,0], tv[:,1], tv[:,2], color='green', alpha=0.12)
    for i, j in [(0,1),(1,2),(2,3),(3,0)]:
        ax.plot3D([tv[i,0],tv[j,0]], [tv[i,1],tv[j,1]], [tv[i,2],tv[j,2]], 'darkgreen', linewidth=1)

    # 轨迹
    t = trajectory[:,0]
    ax.scatter3D(X, Y, Z, c=t, cmap='viridis', s=12, alpha=0.85)
    ax.plot3D(X, Y, Z, 'b-', linewidth=1, alpha=0.35)
    ax.scatter3D(X[0], Y[0], Z[0], c='lime', s=120, marker='o',
                edgecolors='darkgreen', linewidths=2, label='Start')
    ax.scatter3D(X[-1], Y[-1], Z[-1], c='orange', s=100, marker='s',
                edgecolors='darkorange', linewidths=2, label='End')

    # 落点
    if fit:
        c_land = 'limegreen' if fit['in_bounds'] else 'red'
        ax.scatter3D(fit['X_land'], fit['Y_land'], TABLE_H, c=c_land, s=180, marker='X',
                    linewidths=3, label=f"Landing ({'IN' if fit['in_bounds'] else 'OUT'})")
        # 预测轨迹延长线
        t_ext = np.linspace(0, fit['t_land'], 30)
        X_ext = fit['X0'] + fit['Vx0']*t_ext
        Y_ext = fit['Y0'] + fit['Vy0']*t_ext
        Z_ext = fit['Z0'] + fit['Vz0']*t_ext - 0.5*G*t_ext**2
        mask = Z_ext >= TABLE_H
        ax.plot3D(X_ext[mask], Y_ext[mask], Z_ext[mask], 'r--', linewidth=1.5, alpha=0.6, label='Predicted path')

    ax.set_xlabel('X (m)', fontsize=11); ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_zlabel('Z (m)', fontsize=11)
    ax.set_title('3D Trajectory Reconstruction & Landing Prediction', fontsize=14, fontweight='bold')
    ax.set_xlim(-0.3, 3.2); ax.set_ylim(-0.3, 1.8); ax.set_zlim(0, 3.5)
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    f1 = os.path.join(out_dir, '3d_trajectory.png')
    plt.savefig(f1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  {f1}")

    # ============================================================
    # 图2: 俯视落点图
    # ============================================================
    fig, ax = plt.subplots(figsize=(10, 8))

    # 球台
    ax.add_patch(plt.Rectangle((0,0), TABLE_L, TABLE_W, fill=False, edgecolor='green', linewidth=2))
    ax.axvline(x=TABLE_L/2, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)

    # 轨迹投影
    sc = ax.scatter(X, Y, c=t, cmap='viridis', s=15, alpha=0.8, zorder=5)
    ax.plot(X, Y, 'b-', linewidth=1, alpha=0.35, zorder=4)
    ax.scatter(X[0], Y[0], c='lime', s=120, marker='o', edgecolors='darkgreen', linewidths=2, zorder=6, label='Start')

    # 落点
    if fit:
        c_color = 'limegreen' if fit['in_bounds'] else 'red'
        label_text = f"Predicted Landing\n({fit['X_land']:.3f}, {fit['Y_land']:.3f})"
        ax.scatter(fit['X_land'], fit['Y_land'], c=c_color, s=250, marker='X',
                  linewidths=3, edgecolors='black', zorder=10, label=label_text)

        # XY 预测路径
        t_ext_xy = np.linspace(0, fit['t_land'], 50)
        X_xy = fit['X0'] + fit['Vx0']*t_ext_xy
        Y_xy = fit['Y0'] + fit['Vy0']*t_ext_xy
        ax.plot(X_xy, Y_xy, 'r--', linewidth=1, alpha=0.4)

    ax.set_xlabel('X (m)', fontsize=12); ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Landing Point Prediction (Top View)', fontsize=13, fontweight='bold')
    ax.set_xlim(-0.3, 3.2); ax.set_ylim(-0.3, 1.8)
    ax.set_aspect('equal')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(alpha=0.2)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Time (s)', fontsize=10)

    plt.tight_layout()
    f2 = os.path.join(out_dir, 'landing_topview.png')
    plt.savefig(f2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  {f2}")

    # ============================================================
    # 保存数据
    # ============================================================
    np.save(os.path.join(out_dir, 'trajectory_3d.npy'), trajectory)
    np.savetxt(os.path.join(out_dir, 'trajectory_3d.csv'), trajectory,
               delimiter=',', header='t(s),X(m),Y(m),Z(m)', comments='')

    with open(os.path.join(out_dir, 'report.txt'), 'w', encoding='utf-8') as rf:
        rf.write("乒乓球三维轨迹重建与落点预测 — 运行报告\n")
        rf.write("=" * 45 + "\n\n")
        rf.write("【相机参数】\n")
        rf.write(f"  Camera A: 焦距 fx={K_A_c[0,0]:.1f} px\n")
        rf.write(f"  Camera B: 焦距 fx={K_B_c[0,0]:.1f} px\n")
        rf.write(f"  双目基线长度: {np.linalg.norm(T):.4f} m\n\n")
        rf.write("【三维轨迹】\n")
        rf.write(f"  轨迹点数: {len(trajectory)}\n")
        rf.write(f"  X 范围: [{X.min():.2f}, {X.max():.2f}] m\n")
        rf.write(f"  Y 范围: [{Y.min():.2f}, {Y.max():.2f}] m\n")
        rf.write(f"  Z 范围: [{Z.min():.2f}, {Z.max():.2f}] m\n\n")
        if fit:
            rf.write("【轨迹拟合】\n")
            rf.write(f"  决定系数 R2: {fit['r2']:.4f}\n")
            rf.write(f"  均方根误差 RMSE: {fit['rmse']:.4f} m\n")
            rf.write(f"  初始速度: ({fit['Vx0']:.2f}, {fit['Vy0']:.2f}, {fit['Vz0']:.2f}) m/s\n\n")
            rf.write("【落点预测】\n")
            rf.write(f"  预测落点: X = {fit['X_land']:.3f} m, Y = {fit['Y_land']:.3f} m\n")
            rf.write(f"  是否在球台内: {'是' if fit['in_bounds'] else '否（出界）'}\n")
        else:
            rf.write("【轨迹拟合】失败（数据点不足）\n")

    print(f"  trajectory_3d.csv, report.txt")
    print(f"\n完成! 输出: {out_dir}/")


if __name__ == '__main__':
    main()
