import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def rgb2gray(img):
    """
    将RGB图像转换为灰度图像
    """
    img=np.array(img)  # 转换为NumPy数组
    if len(img.shape) == 3 and img.shape[2] >= 3:  # 检查是否为RGB图像
        r_channel=img[:,:,0].astype(np.float64)
        g_channel=img[:,:,1].astype(np.float64)
        b_channel=img[:,:,2].astype(np.float64)

        gray=0.299*r_channel+0.587*g_channel+0.114*b_channel  # 使用加权平均法转换为灰度
        gray=np.clip(gray,0,255).astype(np.uint8)  # 确保灰度值在0-255范围内，并转换为uint8类型
    
    elif len(img.shape)==2:
        gray=img.astype(np.uint8)  # 如果已经是灰度图像，直接转换为uint8类型
    
    else:
        raise ValueError("不支持的图像格式，图像应为RGB或灰度图像")
        
    return gray

def rgb2binary(img,threshold=128):
    """
    将rgb图像转换为二值图像
    """ 
    gray=rgb2gray(img)
    binary=(gray>=threshold).astype(np.uint8)  # 使用阈值将灰度图像转换为二值图像
    
    return binary

def gray2binary(img_gray,threshold=128):
    """
    将灰度图像转换为二值图像
    """
    binary=(img_gray>=threshold).astype(np.uint8)  # 使用阈值将灰度图像转换为二值图像
    
    return binary


if __name__ == "__main__":
    img=Image.open("test1.png") 
    gray_image = rgb2gray(img)
    print(gray_image) 
    plt.imshow(gray_image, cmap='gray')
    plt.axis('off')
    plt.show()