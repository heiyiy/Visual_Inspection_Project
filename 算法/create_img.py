import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 生成模拟的二值图像 
# ==========================================
def create_synthetic_image():
    # 创建一个 10x10 的全白画布 (0 = 白色)
    img = np.zeros((10, 10), dtype=int)

    # 手动绘制左上角的竖条 
    img[2, 2] = 1  # 在第2列，第2行画竖线
    img[2, 1] = 1    # 左边突出一个
    img[2, 3] = 1    # 右边突出一个

    # 手动绘制左下角的方块 
    img[6:9,1]=1
    img[6:9,3]=1
    img[6,2]=1
    img[8,2]=1
    # 手动绘制右侧的T字形 
    img[4:7, 6:9] = 1


    return img

if __name__ == "__main__":
    synthetic_image = create_synthetic_image()
    plt.figure(figsize=(2,2))
    plt.imshow(synthetic_image, cmap='gray_r',interpolation='nearest')  # 使用灰度反转显示，最近邻插值
    plt.xticks([])  # 去掉 x 轴的刻度数字
    plt.yticks([])  # 去掉 y 轴的刻度数字
    plt.savefig("test_large2.png",dpi=100)  # 保存生成的图像为 test.png
    plt.show()
