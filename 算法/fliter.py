import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from img_trans import rgb2gray

# 均值滤波
def mean_filter(image, ksize):
    """
    image : 输入灰度图 (numpy array)
    ksize : 滤波算子大小（奇数）
    """
    assert ksize % 2 == 1, "ksize必须是奇数"
    
    pad = ksize // 2
    padded = np.pad(image, pad, mode='edge')#边界直接复制
    h, w = np.shape(image)
    result = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            window=padded[i:i+ksize, j:j+ksize]
            result[i, j] = np.mean(window)

    return result.astype(np.uint8)

#中值滤波
def median_filter(image, ksize):
    """
    image : 输入灰度图 (numpy array)
    ksize : 滤波算子大小（奇数）
    """
    assert ksize % 2 == 1, "ksize必须是奇数"
    pad = ksize // 2
    padded = np.pad(image, pad, mode='edge')#边界直接复制
    
    h, w = np.shape(image)
    result = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            window = padded[i:i+ksize, j:j+ksize]
            result[i, j] = np.median(window)

    return result

if __name__ == "__main__":
    img=Image.open('stripe_with_border.png')
    img=rgb2gray(img)
    
    # mean_result=mean_filter(img, 3)
    median_result=median_filter(img, 5)
    
    plt.figure(figsize=(8, 4))
    #原图
    plt.subplot(1, 2, 1)    
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    #处理后
    plt.subplot(1, 2, 2)
    plt.imshow(median_result, cmap='gray')
    plt.title("Median 5x5")
    plt.axis('off')
    plt.tight_layout()
    plt.show()