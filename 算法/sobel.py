import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from img_trans import rgb2gray


def sobel(image):
    """
    Sobel算子边缘增强
    image : 输入灰度图 (numpy array, uint8)
    return: Sobel梯度幅值图 (numpy array, uint8)
    """
    # Sobel水平方向卷积核（检测垂直边缘）
    Gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

    # Sobel垂直方向卷积核（检测水平边缘）
    Gy = np.array([[-1, -2, -1],
                   [0,  0,  0],
                   [1,  2,  1]])

    # 对图像进行边缘填充，核大小为3x3，所以填充1像素
    pad = 1
    padded = np.pad(image, pad, mode='edge')  # 边界用边缘值复制

    h, w = np.shape(image)  # 获取原图的高和宽
    result_x = np.zeros((h, w), dtype=np.float32)  # 存放水平方向梯度
    result_y = np.zeros((h, w), dtype=np.float32)  # 存放垂直方向梯度

    # 对每个像素进行卷积
    for i in range(h):
        for j in range(w):
            # 取出当前像素的3x3邻域
            window = padded[i:i+3, j:j+3]

            # 与两个卷积核分别做逐元素相乘再求和
            result_x[i, j] = np.sum(window * Gx)  # 水平梯度
            result_y[i, j] = np.sum(window * Gy)  # 垂直梯度

    # 计算梯度幅值：sqrt(Gx^2 + Gy^2)
    gradient = np.sqrt(result_x ** 2 + result_y ** 2)

    # 归一化到0-255并转为uint8
    gradient = gradient / gradient.max() * 255
    gradient = gradient.astype(np.uint8)

    return gradient


if __name__ == "__main__":
    # 读取图像并转换为灰度
    img = Image.open('rice.png')
    gray = rgb2gray(img)

    # 使用Sobel算子处理
    sobel_result = sobel(gray)

    # 显示原图与结果
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(gray, cmap='gray')
    plt.title("Original")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 2, 2)
    plt.imshow(sobel_result, cmap='gray')
    plt.title("Sobel")
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()
    plt.show()
