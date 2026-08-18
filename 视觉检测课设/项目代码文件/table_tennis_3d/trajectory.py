"""
二维轨迹提取程序
============================================
用法: python trajectory.py

从两个视频分别提取乒乓球的二维运动轨迹，生成多张可视化图。
输出: outputs/2d_trajectory_A/B.png, 2d_overlay_A/B.png,
      outputs/2d_keyframes_A/B.png, detection_A/B_*.jpg
"""
import cv2, numpy as np, os, sys
sys.stdout.reconfigure(encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque

# ============================================================
# 检测参数
# ============================================================
# 黄色球在 HSV 空间中的颜色范围，后续用来提取黄色候选区域。
YELLOW_LOWER = np.array([17, 100, 100], dtype=np.uint8)
YELLOW_UPPER = np.array([35, 255, 255], dtype=np.uint8)
# 连通区域面积范围，用来过滤过小噪点和过大误检目标。
MIN_AREA, MAX_AREA = 12, 400
# 最低圆度，用来约束候选区域形状要接近圆形。
MIN_CIRCULARITY = 0.65
# 最终置信度阈值，低于该值不认为是球。
SCORE_THRESHOLD = 0.55
# 运动一致性阈值：用于给与与上一帧位置相近的候选更高分。
MAX_TRACK_DIST = 120


class BallDetector:
    def __init__(self):
        # 前景检测器，用于提取运动区域。
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=16, detectShadows=False)
        # 保存最近 3 帧灰度图，用于时间差分运动检测。
        self.prev_frames = deque(maxlen=3)
        # 形态学核，用于开闭运算，去噪和填充小空洞。
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # 记录上一帧的球心，用于运动一致性评分。
        self.last_center = None
        # 连续漏检计数，避免短暂丢帧后位置跟踪失败。
        self.missed_frames = 0

    def preprocess(self, frame):
        # 对输入图像做高斯模糊，减少噪点影响。
        return cv2.GaussianBlur(frame, (5, 5), 1.0)

    def detect_motion(self, frame):
        # 背景减除得到运动前景，并做一次开运算去噪。
        fg = cv2.morphologyEx(self.bg_sub.apply(frame), cv2.MORPH_OPEN, self.kernel)
        # 时间差分用于检测持续运动区域。
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_frames.append(gray)
        diff = np.zeros_like(fg)
        if len(self.prev_frames) >= 3:
            f2, f1, f0 = list(self.prev_frames)
            d1 = cv2.threshold(cv2.absdiff(f2, f1), 25, 255, cv2.THRESH_BINARY)[1]
            d2 = cv2.threshold(cv2.absdiff(f1, f0), 25, 255, cv2.THRESH_BINARY)[1]
            diff = cv2.bitwise_and(d1, d2)
        # 将背景减除结果与时间差分结果合并，并膨胀以强化目标区域。
        return cv2.dilate(cv2.bitwise_or(fg, diff), self.kernel, iterations=1)

    def detect_color(self, frame):
        # 将 RGB 图像转为 HSV，并提取黄色区域掩码。
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)

    def detect(self, frame):
        # 对当前帧执行预处理、运动检测和颜色检测。
        processed = self.preprocess(frame)
        motion = self.detect_motion(processed)
        color = self.detect_color(processed)

        # 运动与颜色同时满足的区域更可能是乒乓球。
        candidate = cv2.bitwise_and(motion, color)
        if cv2.countNonZero(candidate) < 5:
            # 如果交集区域太小，说明运动和颜色不够一致。
            # 如果颜色区域本身也很少，就退回到颜色区域；否则直接放弃。
            if cv2.countNonZero(color) < 200:
                candidate = color
            else:
                return None, 0.0

        hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate, 8)
        best_score, best_center = -1, None
        best_dist = None

        for i in range(1, n):
            # 按连通域过滤候选区域。
            area = stats[i, cv2.CC_STAT_AREA]
            if area < MIN_AREA or area > MAX_AREA:
                continue

            comp = (labels == i).astype(np.uint8)
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)

            peri = cv2.arcLength(cnt, True)
            if peri < 1:
                continue
            circ = 4 * np.pi * area / (peri * peri)
            if circ < MIN_CIRCULARITY:
                continue

            x, y, w_box, h_box, _ = stats[i]
            aspect = min(w_box, h_box) / max(w_box, h_box)
            if aspect < 0.65:
                continue

            # 进一步检查区域的 HSV 平均值，确保它确实是“黄色"。
            mask = comp.astype(bool)
            mean_hue = hsv[:,:,0][mask].mean() if mask.any() else 0
            mean_sat = hsv[:,:,1][mask].mean() if mask.any() else 0
            mean_val = hsv[:,:,2][mask].mean() if mask.any() else 0
            if mean_sat < 110 or mean_val < 120:
                continue

            # 面积、圆度、长宽比综合得分。
            area_s = 1.0 if 15 < area < 250 else max(0, 1 - abs(area-70)/200)
            shape_score = 0.35 * area_s + 0.55 * circ + 0.10 * aspect

            # 重新计算更精确的中心点，避免连通域质心偏移。
            cx, cy = centroids[i]
            x0, y0 = max(0, int(cx)-3), max(0, int(cy)-3)
            x1, y1 = min(processed.shape[1], int(cx)+4), min(processed.shape[0], int(cy)+4)
            roi = cv2.cvtColor(processed[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            if roi.sum() > 0:
                cx = (xx * roi).sum() / roi.sum()
                cy = (yy * roi).sum() / roi.sum()

            # 运动一致性：距离上一帧目标越近得分越高。
            dist = float('inf')
            if self.last_center is not None:
                dist = np.hypot(cx - self.last_center[0], cy - self.last_center[1])
            dist_score = 0.0
            if dist < MAX_TRACK_DIST:
                dist_score = max(0.0, 1.0 - dist / MAX_TRACK_DIST)

            score = shape_score + 0.25 * dist_score
            if score > best_score:
                best_score = score
                best_center = (float(cx), float(cy))
                best_dist = dist

        # 如果找到一个足够好的候选，则返回它。
        if best_center is not None and best_score > SCORE_THRESHOLD:
            if best_dist is not None and best_dist > MAX_TRACK_DIST and self.last_center is not None:
                self.missed_frames += 1
            else:
                self.missed_frames = 0
            if self.missed_frames > 5:
                self.last_center = None
                self.missed_frames = 0
            else:
                self.last_center = best_center
            return best_center, best_score

        # 如果未检测到，增加漏检计数，短暂允许连续丢帧。
        if self.missed_frames > 0:
            self.missed_frames += 1
            if self.missed_frames > 5:
                self.last_center = None
                self.missed_frames = 0
        return None, 0.0


def process_video(video_path, label, out_dir):
    # 打开视频并读取基本信息。
    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(5)
    total = int(cap.get(7))

    # 从第 10 帧读取一帧作为背景参考，用于可视化底图。
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(10, total-1))
    ret, bg = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    detector = BallDetector()
    detections = []  # 保存检测结果: (frame_idx, x, y, conf)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        c, conf = detector.detect(frame)
        # 只保存置信度高于 0.5 的检测结果。
        if c is not None and conf > 0.5:
            detections.append((frame_idx, c[0], c[1], conf))
        frame_idx += 1
    cap.release()

    n_det = len(detections)
    print(f"  Camera {label}: {n_det} / {total} frames ({100*n_det/total:.1f}%)")

    if n_det < 3:
        print(f"    Not enough detections!")
        return

    pts = np.array([(d[1], d[2]) for d in detections])
    times = [d[0]/fps for d in detections]
  
    # 图1: 2D像素轨迹 + X/Y时序
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    ax1.plot(pts[:,0], pts[:,1], 'b-', linewidth=1, alpha=0.4)
    sc = ax1.scatter(pts[:,0], pts[:,1], c=times, s=15, cmap='viridis')
    ax1.scatter(*pts[0], s=120, c='lime', marker='o', edgecolors='darkgreen', linewidths=2, label=f'Start (t={times[0]:.2f}s)')
    ax1.scatter(*pts[-1], s=120, c='red', marker='X', linewidths=2, label=f'End (t={times[-1]:.2f}s)')
    ax1.set_xlim(0, w); ax1.set_ylim(h, 0)
    ax1.set_xlabel('X (pixels)'); ax1.set_ylabel('Y (pixels)')
    ax1.set_title(f'Camera {label} - 2D Pixel Trajectory ({n_det} points)', fontweight='bold')
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    plt.colorbar(sc, ax=ax1, label='Time (s)')

    ax2.plot(times, pts[:,0], 'r-', linewidth=1.5, label='X (horizontal)')
    ax2.plot(times, pts[:,1], 'b-', linewidth=1.5, label='Y (vertical)')
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Pixel position')
    ax2.set_title('X/Y Position vs Time', fontweight='bold')
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.suptitle(f'Camera {label} - 2D Ball Motion Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    f1 = os.path.join(out_dir, f'2d_trajectory_{label}.png')
    plt.savefig(f1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    {f1}")

    # 关键帧拼贴图：将若干关键帧与其前后局部轨迹一起展示。
   
    # 图3: 关键帧拼贴（每帧上叠加局部轨迹）
    n_key = 6
    interval = max(1, total // n_key)
    key_frames = list(range(0, total, interval))[:n_key]

    cap2 = cv2.VideoCapture(video_path)
    frames_data = []
    for fid in key_frames:
        cap2.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frm = cap2.read()
        if ret: frames_data.append((fid, frm))
    cap2.release()

    nf = len(frames_data)
    cols = min(3, nf)
    rows = (nf + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if rows == 1 and cols == 1: axes = np.array([[axes]])
    elif rows == 1: axes = axes.reshape(1, -1)
    elif cols == 1: axes = axes.reshape(-1, 1)

    for i, (fid, frm) in enumerate(frames_data):
        r, c = i // cols, i % cols
        ax = axes[r, c]
        ax.imshow(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        # 只显示当前关键帧附近的检测点，便于定位该帧的局部轨迹。
        window = max(5, total // 30)
        nearby = sorted([d for d in detections if abs(d[0]-fid) <= window], key=lambda x: x[0])
        if nearby:
            nx = [d[1] for d in nearby]; ny = [d[2] for d in nearby]
            ax.plot(nx, ny, 'y-', linewidth=2, alpha=0.8)
            ax.scatter(nx, ny, c=[d[3] for d in nearby], s=25, cmap='hot',
                      edgecolors='white', linewidth=0.3, zorder=10)
        # 当前帧的球
        cur = [d for d in detections if d[0] == fid]
        if cur:
            ax.scatter([cur[0][1]], [cur[0][2]], s=200, marker='o',
                      facecolors='none', edgecolors='lime', linewidth=3)
        ax.set_title(f'Frame {fid}', fontsize=10)
        ax.set_xlim(0, w); ax.set_ylim(h, 0)
        ax.axis('off')

    for i in range(nf, rows*cols):
        axes[i//cols, i%cols].axis('off')

    plt.suptitle(f'Camera {label} - 2D Trajectory on Key Frames', fontsize=13, fontweight='bold')
    plt.tight_layout()
    f3 = os.path.join(out_dir, f'2d_keyframes_{label}.png')
    plt.savefig(f3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    {f3}")

    # 图4-6: 球检测帧（选3张最好的）
    # 从检测结果中挑选“最好”的三帧进行可视化。
    # 这里优先选择更靠近画面中心的点，以减少边缘假阳性。
    best = []
    for fid, cx, cy, conf in detections:
        cn = abs(cx/w - 0.5); yn = abs(cy/h - 0.5)
        best.append((np.sqrt(cn**2+yn**2), conf, fid))
    best.sort()

    cap3 = cv2.VideoCapture(video_path)
    for i, (_, _, fid) in enumerate(best[:3]):
        cap3.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frm = cap3.read()
        if not ret: continue
        # 找到该帧的检测
        frame_dets = [d for d in detections if d[0] == fid]
        if frame_dets:
            _, cx, cy, conf = frame_dets[0]
            vis = frm.copy()
            cv2.circle(vis, (int(cx), int(cy)), 25, (0, 255, 255), 3)
            cv2.circle(vis, (int(cx), int(cy)), 6, (0, 255, 255), -1)
            cv2.line(vis, (int(cx)-35, int(cy)), (int(cx)+35, int(cy)), (0,255,255), 1)
            cv2.line(vis, (int(cx), int(cy)-35), (int(cx), int(cy)+35), (0,255,255), 1)
            cv2.putText(vis, 'Ping-Pong Ball', (int(cx)+30, int(cy)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            cv2.putText(vis, f'Conf: {conf:.2f} | Frame: {fid}',
                       (int(cx)+30, int(cy)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            cv2.putText(vis, f'Camera {label}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            f4 = os.path.join(out_dir, f'detection_{label}_{i}.jpg')
            _, buf = cv2.imencode('.jpg', vis)
            with open(f4, 'wb') as wf: wf.write(buf)
            print(f"    {f4} (frame {fid}, conf={conf:.2f})")
    cap3.release()


def make_single_overlay(video_path, label, out_dir):
    """单相机轨迹叠加图：在背景帧上绘制完整轨迹，并标记时间点、起点和终点。"""
    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(5)
    total = int(cap.get(7))
    detector = BallDetector()

    cap.set(cv2.CAP_PROP_POS_FRAMES, min(10, total-1))
    ret, bg = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    detections, frame_idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        c, conf = detector.detect(frame)
        if c is not None and conf > 0.5:
            detections.append((frame_idx, c[0], c[1], conf))
        frame_idx += 1
    cap.release()

    if len(detections) < 3:
        return None
    # 将检测点转换为坐标和时间序列，用于绘图。
    pts = np.array([(d[1], d[2]) for d in detections])
    times = [d[0]/fps for d in detections]

    fig, ax = plt.subplots(figsize=(14, 8))
    if bg is not None:
        ax.imshow(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))

    for i in range(1, len(pts)):
        ax.plot(pts[i-1:i+1,0], pts[i-1:i+1,1], '-',
               color=plt.cm.viridis(i/len(pts)), linewidth=2.5, alpha=0.8)
    ax.scatter(*pts[0], s=250, c='lime', marker='o', edgecolors='darkgreen', linewidths=3, zorder=10)
    ax.scatter(*pts[-1], s=250, c='red', marker='X', linewidths=4, zorder=10)

    for j in [len(pts)//4, len(pts)//2, 3*len(pts)//4]:
        ax.annotate(f'{times[j]:.1f}s', (pts[j,0]+20, pts[j,1]-15),
                   fontsize=10, color='white', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.set_title(f'Camera {label} — Ball Trajectory ({len(detections)} pts, {times[-1]-times[0]:.1f}s)',
                fontsize=13, fontweight='bold')
    ax.axis('off')
    sm = plt.cm.ScalarMappable(cmap='viridis'); sm.set_array(times)
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Time (s)', fontsize=10)

    save_path = os.path.join(out_dir, f'trajectory_overlay_{label}.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    {save_path}")
    return detections


def get_trajectory_data(video_path):
    """提取检测结果数据，用于生成双相机并排对比图。"""
    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(5)
    total = int(cap.get(7))
    detector = BallDetector()

    cap.set(cv2.CAP_PROP_POS_FRAMES, min(10, total-1))
    ret, bg = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    detections, frame_idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        c, conf = detector.detect(frame)
        if c is not None and conf > 0.5:
            detections.append((frame_idx, c[0], c[1], conf))
        frame_idx += 1
    cap.release()

    if len(detections) < 3 or bg is None:
        return None
    pts = np.array([(d[1], d[2]) for d in detections])
    times = [d[0]/fps for d in detections]
    return {
        'bg': bg,
        'pts': pts,
        'times': times,
        'w': w,
        'h': h,
        'n_pts': len(detections),
        'duration': times[-1]-times[0]
    }


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base, 'outputs')
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 50)
    print("  二维轨迹提取")
    print("=" * 50)

    # 先分别处理 A、B 两个相机视频，生成各自轨迹图和关键帧图。
    for label in ['A', 'B']:
        vpath = os.path.join(base, 'data', 'videos', f'{label}.mp4')
        print(f"\nCamera {label}:")
        process_video(vpath, label, out_dir)
        make_single_overlay(vpath, label, out_dir)

    # 双相机并排对比
    print("\n生成双相机对比图...")
    data = {}
    for label in ['A', 'B']:
        data[label] = get_trajectory_data(os.path.join(base, 'data', 'videos', f'{label}.mp4'))

    # 只有当 A 和 B 都有有效轨迹时，才绘制并排对比图。
    if data['A'] is not None and data['B'] is not None:
        fig, axes = plt.subplots(1, 2, figsize=(20, 7.5))
        for ax, label in zip(axes, ['A', 'B']):
            d = data[label]
            ax.imshow(cv2.cvtColor(d['bg'], cv2.COLOR_BGR2RGB))
            pts, times = d['pts'], d['times']
            for i in range(1, len(pts)):
                ax.plot(pts[i-1:i+1,0], pts[i-1:i+1,1], '-',
                       color=plt.cm.viridis(i/len(pts)), linewidth=2.5, alpha=0.8)
            ax.scatter(*pts[0], s=220, c='lime', marker='o', edgecolors='darkgreen', linewidths=3)
            ax.scatter(*pts[-1], s=220, c='red', marker='X', linewidths=4)
            for j in [len(pts)//4, len(pts)//2, 3*len(pts)//4]:
                ax.annotate(f'{times[j]:.1f}s', (pts[j,0]+15, pts[j,1]-10),
                           fontsize=9, color='white', fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            ax.set_xlim(0, d['w']); ax.set_ylim(d['h'], 0)
            ax.set_title(f'Camera {label} ({d["n_pts"]} 个检测点, {d["duration"]:.1f}秒)',
                        fontsize=12, fontweight='bold')
            ax.axis('off')
        plt.suptitle('2D Ball Trajectory — Dual Camera Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'trajectory_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  trajectory_comparison.png")

    plt.close('all')
    print(f"\n完成! 输出目录: {out_dir}/")


if __name__ == '__main__':
    main()
