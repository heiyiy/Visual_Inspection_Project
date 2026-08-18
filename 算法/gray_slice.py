import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from img_trans import rgb2gray

def bit_plane_slice(img_gray):
    '''
    位面切割
    '''
    planes=[]#空列表
    plt.figure(figsize=(12,8))#宽12高8
    
    for i in range(8):#针对8bit图像
        plane=(img_gray>>i)&1#★★★注意：>>右移运算，把每个像素值都同时右移i位，&1按位与运算。结果得到0-7位面
        planes.append(plane.astype(np.uint8)*255)#转为255
        plt.subplot(3,3,i+1)
        plt.imshow(planes[i],cmap='gray')
        plt.title(f"bit plane {i}")
        plt.axis('off')
        
    plt.subplot(3,3,9)
    plt.imshow(img_gray,cmap='gray')
    plt.title(f'original img')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    return planes
    
if __name__=="__main__":
    img_gray=rgb2gray(Image.open("test5.jpeg"))
    bit_plane_slice(img_gray)
    

    
    
   