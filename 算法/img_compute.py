import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from img_trans import rgb2gray,rgb2binary

def add(img1,img2):
    img_add=(img1+img2).clip(0,255)  #拉回0-255
    
    return img_add

def subtract(img1,img2):
    img_subtract=(img1-img2)
    img_subtract[img_subtract<0]=0  # 将负值设为0
        
    return img_subtract

def andd(img1,img2):
    img_and=(img1&img2)
        
    return img_and

def orr(img1,img2):
    img_or=(img1|img2)
    
    return img_or

def xor(img1,img2):
    img_xor=(img1^img2)

    return img_xor
    
    
if __name__ == "__main__":
    img1,img2=Image.open('test1.png'),Image.open('test4.png')
    # img2=img2.resize(img1.size)#改大小
    img1_gray,img2_gray=rgb2gray(img1),rgb2gray(img2)
    img1_binary,img2_binary=rgb2binary(img1),rgb2binary(img2)
    add=add(img1_gray,img2_gray)
    subtract=subtract(img1_gray,img2_gray)
    andd=andd(img1_binary,img2_binary)
    orr=orr(img1_binary,img2_binary)
    xor=xor(img1_binary,img2_binary)

    plt.figure(figsize=(15, 3))
    
    # 原始图像1
    plt.subplot(1, 7, 1)
    plt.imshow(img1_gray, cmap='gray')
    plt.title("Original Image 1")
    plt.xticks([])
    plt.yticks([])
    
    # 原始图像2
    plt.subplot(1, 7, 2)
    plt.imshow(img2_gray, cmap='gray')
    plt.title("Original Image 2")
    plt.xticks([])
    plt.yticks([])
    
    # 相加
    plt.subplot(1, 7, 3)
    # 注意：相加结果可能有值2，所以需要调整显示范围
    plt.imshow(add, cmap='gray', vmin=0, vmax=2)
    plt.title("Img1+img2")
    plt.xticks([])
    plt.yticks([])
    
    # 相减
    plt.subplot(1, 7, 4)
    plt.imshow(subtract, cmap='gray')
    plt.title("Img1-Img2")
    plt.xticks([])
    plt.yticks([])
    
    # 与运算
    plt.subplot(1, 7, 5)
    plt.imshow(andd, cmap='gray')
    plt.title("Img1&Img2")
    plt.xticks([])
    plt.yticks([])
    
    # 或运算
    plt.subplot(1, 7, 6)
    plt.imshow(orr, cmap='gray')
    plt.title("Img1|Img2")
    plt.xticks([])
    plt.yticks([])
    
    # 异或
    plt.subplot(1, 7, 7)
    plt.imshow(xor, cmap='gray')
    plt.title("Img1^Img2")
    plt.xticks([])
    plt.yticks([])
    
    plt.tight_layout()
    plt.show()
    