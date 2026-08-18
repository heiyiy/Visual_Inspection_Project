import cv2, numpy as np, os, sys, yaml
sys.stdout.reconfigure(encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


#棋盘格参数
BOARD_COLS, BOARD_ROWS=5, 8 #内角点数
SQUARE_SIZE=0.028  #每个方格边长度（米）

#标定函数
def calibrate():
    base=os.path.dirname(os.path.abspath(__file__))
    dir_a=os.path.join(base, 'data', 'calibration', 'camera_A')
    dir_b=os.path.join(base, 'data', 'calibration', 'camera_B')
    out_dir=os.path.join(base, 'outputs')
    calib_file=os.path.join(base, 'calibration.yaml')
    os.makedirs(out_dir, exist_ok=True)

    #世界坐标点生成
    objp=np.zeros((BOARD_COLS * BOARD_ROWS, 3), np.float32)#创建一个5*8共40个三维点的空数组，每个点有（x，y，z）
    #生成5*8网格坐标，转置后变成40行2列的数组，再乘以实际边长，得到角点在世界坐标系的位置，z为0
    objp[:, :2]=np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2)*SQUARE_SIZE

    #单目标定
    def calib_one(img_dir):
        files=sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg','.png','.jpeg'))])
        obj_pts, img_pts, corners_list=[], [], []#储存世界坐标，图像坐标
        img_size=None
        for f in files:
            img=cv2.imdecode(np.fromfile(os.path.join(img_dir, f), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None: continue
            if img_size is None: img_size=(img.shape[1], img.shape[0])#记录图像宽和高，OpenCV直接shape是 (高,宽,通道)
            gray=cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))#彩色转灰度，直方图均衡化
            ret, corners=cv2.findChessboardCorners(gray, (BOARD_COLS, BOARD_ROWS),
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)#ret是否检测到棋盘格，corners角点坐标（40，1，2）
            if ret:
                #亚像素级角点优化，窗口大小11*11，停止条件：30次迭代或误差小于0.001
                corners=cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                obj_pts.append(objp)
                img_pts.append(corners)
                corners_list.append((f, img, corners))
        if len(obj_pts) < 3:
            raise RuntimeError(f"标定图不足: {len(obj_pts)}")#每张图像提供 2 个约束，内参有 5 个自由度需要求解，至少需要3张图
        #执行张正友标定，返回内参矩阵K和畸变系数dist
        _, K, dist, _, _=cv2.calibrateCamera(obj_pts, img_pts, img_size, None, None)#第一个None：不提供初始内参矩阵，让算法从零估计，第二个None：不固定任何内参
        return K, dist, img_size, len(obj_pts), corners_list

    print("进行相机标定")

    print("\n标定 Camera A")
    K_A, dist_A, size_A, n_A, corners_A = calib_one(dir_a)
    print(f"  使用 {n_A} 张图像, 分辨率 {size_A[0]}x{size_A[1]}")
    print(f"  fx={K_A[0,0]:.1f}, fy={K_A[1,1]:.1f}, cx={K_A[0,2]:.1f}, cy={K_A[1,2]:.1f}")
    #fx：水平焦距（以像素为单位）和fy：垂直焦距，主点(cx,cy)
    print("\n标定 Camera B")
    K_B, dist_B, size_B, n_B, corners_B = calib_one(dir_b)
    print(f"  使用 {n_B} 张图像, 分辨率 {size_B[0]}x{size_B[1]}")
    print(f"  fx={K_B[0,0]:.1f}, fy={K_B[1,1]:.1f}, cx={K_B[0,2]:.1f}, cy={K_B[1,2]:.1f}")



    #立体标定: 遍历所有候选同步对，用 solvePnP 求解并评估，选立体误差最小的那对
    print("\n立体标定 (solvePnP, 选最优同步对)...")
    corners_map_A={fa: ca for fa, _, ca in corners_A}
    corners_map_B={fb: cb for fb, _, cb in corners_B}
    sync_files=sorted(set(corners_map_A.keys()) & set(corners_map_B.keys()))
    print(f"  找到 {len(sync_files)} 对候选同步图像")

    if len(sync_files) < 1:
        raise RuntimeError("找不到同步棋盘格对!")

    #逐对评估
    pair_results=[]
    print(f"  {'文件':>6s}  {'A单目err':>10s}  {'B单目err':>10s}  {'stereo err':>11s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*11}")
    for f in sync_files:
        ca, cb=corners_map_A[f], corners_map_B[f]

        #各自solvePnP求解物体到相机的位姿：rA、tA是物体坐标系到相机A坐标系的旋转向量和平移向量
        _, rA, tA=cv2.solvePnP(objp, ca, K_A, dist_A)
        _, rB, tB=cv2.solvePnP(objp, cb, K_B, dist_B)

        #求单目重投影误差，将世界坐标系的角点通过求得的位姿投影回图像平面，计算与实际检测到的角点位置的平均距离作为误差指标
        projA, _=cv2.projectPoints(objp, rA, tA, K_A, dist_A)
        projB, _=cv2.projectPoints(objp, rB, tB, K_B, dist_B)
        errA=np.mean(np.sqrt(np.sum((ca - projA)**2, axis=2)))
        errB=np.mean(np.sqrt(np.sum((cb - projB)**2, axis=2)))

        #计算 AB相机坐标系之间的R, T
        RA, _=cv2.Rodrigues(rA)
        RB, _=cv2.Rodrigues(rB)
        R_pair=RA @ RB.T
        T_pair=tA - R_pair @ tB

        #三角化+重投影评估stereo一致性
        P1=K_A @ np.hstack([np.eye(3), np.zeros((3, 1))])#相机a的投影矩阵，假设相机a坐标系为世界坐标系，所以旋转矩阵是单位矩阵，平移向量是零向量
        P2=K_B @ np.hstack([R_pair, T_pair.reshape(3, 1)])#相机b的投影矩阵，将相机b坐标系转换到相机a坐标系下
        pts4d=cv2.triangulatePoints(P1, P2, ca.reshape(-1, 2).T, cb.reshape(-1, 2).T)#三角化：根据两台相机的投影矩阵和对应的角点，计算出这些点在三维空间中的位置
        pts3d=(pts4d[:3] / pts4d[3]).T

        #恢复的三d点再重投影回A图像平面，计算与实际角点位置的平均距离作为stereo误差指标
        rvec0=np.zeros(3)
        projA_stereo, _=cv2.projectPoints(pts3d, rvec0, np.zeros(3), K_A, dist_A)
        err_stereo_A=np.mean(np.sqrt(np.sum((ca - projA_stereo)**2, axis=2)))

        # 重投影回 B，同样计算stereo误差
        pts3d_h=np.hstack([pts3d, np.ones((pts3d.shape[0], 1))])
        projB_h=(P2 @ pts3d_h.T).T
        projB_stereo=(projB_h[:, :2] / projB_h[:, 2:3]).reshape(-1, 1, 2)
        err_stereo_B=np.mean(np.sqrt(np.sum((cb - projB_stereo)**2, axis=2)))

        err_stereo=(err_stereo_A + err_stereo_B) / 2#求平均
        pair_results.append((f, err_stereo, R_pair, T_pair, errA, errB))
        print(f"  {f:>6s}  {errA:>10.4f}  {errB:>10.4f}  {err_stereo:>11.4f}")

    #选stereo误差最小的对得到最终的R、T
    best=min(pair_results, key=lambda x: x[1])
    best_file, stereo_err, R, T, _, _=best
    baseline = np.linalg.norm(T)
    print(f"\n  最优同步对: {best_file}")
    print(f"  Stereo 误差: {stereo_err:.4f} px")
    print(f"  Baseline: {baseline:.3f}m")

    # 保存
    calib_data={
        'K_A': K_A.tolist(), 'dist_A': dist_A.ravel().tolist(),
        'K_B': K_B.tolist(), 'dist_B': dist_B.ravel().tolist(),
        'R': R.tolist(), 'T': T.tolist(),
        'image_size_A': size_A, 'image_size_B': size_B,
        'stereo_error': float(stereo_err),
        'best_pair': best_file
    }
    with open(calib_file, 'w', encoding='utf-8') as f:
        yaml.dump(calib_data, f, default_flow_style=False)
    print(f"  已保存: {calib_file}")

    #生成标定结果展示图
    print("\n生成标定展示图...")
    fig=plt.figure(figsize=(16, 10))

    for i, (K, dist, corners_list, label) in enumerate([
        (K_A, dist_A, corners_A, 'A'), (K_B, dist_B, corners_B, 'B')
    ]):
        # 棋盘格角点检测展示
        ax=fig.add_subplot(2, 3, i+1)
        if corners_list:
            fname, img, corners=corners_list[0]
            vis=img.copy()
            cv2.drawChessboardCorners(vis, (BOARD_COLS, BOARD_ROWS), corners, True)
            ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            ax.set_title(f'Camera {label} - Checkerboard Detection', fontweight='bold')
        ax.axis('off')

        #重投影误差计算
        errs=[]
        for fname, img, corners in corners_list:
            _, rvec, tvec=cv2.solvePnP(objp, corners, K, dist)
            proj, _=cv2.projectPoints(objp, rvec, tvec, K, dist)#重投影：将世界坐标系中的3D角点通过相机模型投影回2D图像平面。
            err=np.mean(np.sqrt(np.sum((corners - proj)**2, axis=2)))#计算角点的重投影误差：每个角点的重投影位置与实际检测到的位置之间的欧氏距离，取平均值作为该图像的误差指标。
            errs.append(err)

        ax=fig.add_subplot(2, 3, i+4)
        if errs:
            ax.bar(range(len(errs)), errs, color='steelblue', edgecolor='navy')
            mean_err = np.mean(errs)
            ax.axhline(y=mean_err, color='red', linestyle='--',
                      label=f'Mean: {mean_err:.3f} px')
            ax.set_xlabel('Image #')
            ax.set_ylabel('Reprojection Error (px)')
            ax.set_title(f'Camera {label} - Reprojection Error', fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(axis='y', alpha=0.3)

    # 参数汇总
    ax=fig.add_subplot(2, 3, 6)
    ax.axis('off')
    info_text = (
        "CALIBRATION RESULTS\n"
        "==================\n\n"
        f"Camera A ({size_A[0]}x{size_A[1]}):\n"
        f"  fx = {K_A[0,0]:.1f}  fy = {K_A[1,1]:.1f}\n"
        f"  cx = {K_A[0,2]:.1f}  cy = {K_A[1,2]:.1f}\n"
        f"  k1 = {dist_A.ravel()[0]:.4f}  k2 = {dist_A.ravel()[1]:.4f}\n"
        f"  Images: {n_A}\n\n"
        f"Camera B ({size_B[0]}x{size_B[1]}):\n"
        f"  fx = {K_B[0,0]:.1f}  fy = {K_B[1,1]:.1f}\n"
        f"  cx = {K_B[0,2]:.1f}  cy = {K_B[1,2]:.1f}\n"
        f"  k1 = {dist_B.ravel()[0]:.4f}  k2 = {dist_B.ravel()[1]:.4f}\n"
        f"  Images: {n_B}\n\n"
        f"Stereo (B -> A):\n"
        f"  Best pair = {best_file}\n"
        f"  Stereo error = {stereo_err:.4f} px\n"
        f"  Baseline = {baseline:.4f} m"
    )
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Camera Calibration Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    calib_img = os.path.join(out_dir, 'calibration_results.png')
    plt.savefig(calib_img, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {calib_img}")

    print(f"\n标定完成! 输出文件: calibration.yaml, outputs/calibration_results.png")


if __name__ == '__main__':
    calibrate()
