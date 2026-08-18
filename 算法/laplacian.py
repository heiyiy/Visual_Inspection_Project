import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from img_trans import rgb2gray

## 二阶微分的图像增强-拉普拉斯算子
def laplacian_sharpen(image, mode='4',zhong='zheng'):

    h, w = np.shape(image)

    #定义卷积核
    if mode=='4':
        if zhong=='fu':
            # 四邻域,中心为负
            kernel=np.array([[0,  1, 0],
                            [1, -4, 1],
                            [0,  1, 0]])
        else:
            # 四邻域,中心为正
            kernel=np.array([[0,  -1, 0],
                            [-1, 4, -1],
                            [0,  -1, 0]])
    else:
        if zhong=='fu':
            # 八邻域,中心为负
            kernel=np.array([[1,  1, 1],
                            [1, -8, 1],
                            [1,  1, 1]])
        else:
             # 八邻域,中心为正
            kernel=np.array([[-1,  -1, -1],
                            [-1, 8, -1],
                            [-1,  -1, -1]])
   
    #边界填充
    pad=1
    padded_img=np.pad(image, pad, mode='edge')

    #卷积计算
    lapu=np.zeros((h, w), dtype=np.float32)#创建空数组存储卷积结果
    for i in range(h):
        for j in range(w):
            a=padded_img[i:i+3, j:j+3]# 取出3x3邻域
            lapu[i, j]=np.sum(a*kernel)# ps:填充后的大小为（h+2，w+2），所以i和j的范围是0到h-1和0到w-1,计算时正好对应中心点

    # 原图和lapu叠加
    if zhong=='zheng':
        out=image.astype(np.float32)+lapu
    else:
        out=image.astype(np.float32)-lapu
    out_img=np.clip(out, 0, 255).astype(np.uint8)# 拉回0-255

    # lapu绝对值后归一化
    lapu= np.abs(lapu)
    out_lapu= lapu / lapu.max() * 255#线性归一化到0-255
    out_lapu= out_lapu.astype(np.uint8)

    return out_lapu, out_img


if __name__ == "__main__":
    # 读取图像并转换为灰度
    img = Image.open('moon.jpg')
    gray = rgb2gray(img)

    # 四邻域
    out_lapu4, out_img4 = laplacian_sharpen(gray, mode='4',zhong='fu')
    # 八邻域拉普拉斯增强,
    out_lapu8, out_img8 = laplacian_sharpen(gray, mode='8',zhong='fu')


    plt.figure(figsize=(12, 6))

    plt.subplot(2, 3, 1)
    plt.imshow(gray, cmap='gray')
    plt.title("Original")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 3, 2)
    plt.imshow(out_lapu4, cmap='gray')
    plt.title("out_lapu4")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 3, 3)
    plt.imshow(out_img4, cmap='gray')
    plt.title("out_img4")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 3, 5)
    plt.imshow(out_lapu8, cmap='gray')
    plt.title("out_lapu8")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 3, 6)
    plt.imshow(out_img8, cmap='gray')
    plt.title("out_img8")
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()
    plt.show()
