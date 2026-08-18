import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from img_trans import rgb2gray

##退化函数,h(x,y)=exp(-sqrt(x²+y²)/240)
def build_H(h,w):
    
    #创建坐标网格,以中心为原点
    yy,xx=np.meshgrid(np.arange(h),np.arange(w),indexing='ij')
    yy=yy.astype(np.float32)-h//2  #行坐标，中心为0
    xx=xx.astype(np.float32)-w//2  #列坐标，中心为0

    #h(x,y) = exp(-sqrt(x²+y²)/240)
    dist=np.sqrt(xx**2+yy**2)
    h=np.exp(-dist/10.0)
    h=h/np.sum(h)#归一化

    H=np.fft.fft2(np.fft.ifftshift(h))#把中心原点移到四角，再傅里叶.fft默认原点时[0,0]
    
    return H

##模糊=F*H+N
def build_g(image,H,noise=5):

    h, w=np.shape(image)

    F=np.fft.fft2(image.astype(np.float32))
    fh=np.real(np.fft.ifft2(F * H))#频域相乘F*H,返回空间域
    noise=np.random.normal(0, noise, (h, w)).astype(np.float32)
    g=fh+noise#添加高斯噪声
    g=np.clip(g, 0, 255).astype(np.uint8)#拉回0-255

    return g

##维纳滤波
def weina(image,K=0.01):

    h,w=np.shape(image)

    #调用退化函数
    H=build_H(h, w)
    g=build_g(image, H)

    #维纳滤波:F_hat=G×H*/(|H|²+K)
    G=np.fft.fft2(g.astype(np.float32))
    H_conj=np.conj(H)#H的共轭
    F_pre=G*H_conj/(np.abs(H)**2+K)
    f_pre=np.real(np.fft.ifft2(F_pre))#返回空间域
    f_pre=np.clip(f_pre,0,255).astype(np.uint8)#拉回0-255

    return g,f_pre

##约束最小二乘方滤波,r正则化参数,越大越平滑,也越模糊
def min_ercheng(image, r=0.01):

    h,w=np.shape(image)

    #调用退化函数
    H=build_H(h,w)
    g=build_g(image,H)

    #空域拉普拉斯核四邻域
    l=np.zeros((h, w),dtype=np.float32)
    #将3x3核的5个点放到 (0,0) 附近，ifftshift 后会移到中心
    l[0,0]=-4.0#中心
    l[0,1]=1.0#右
    l[0,w-1]=1.0#左
    l[1,0]=1.0#下
    l[h-1,0]=1.0#上
    #傅里叶变换
    L=np.fft.fft2(np.fft.ifftshift(l))

    #约束最小二乘方滤波:F_hat=G×H*/(|H|² + r|L|²)
    G=np.fft.fft2(g.astype(np.float32))
    H_conj = np.conj(H) #H的共轭
    F_pre = G * H_conj / (np.abs(H)**2+r*(np.abs(L)**2))

    f_pre=np.real(np.fft.ifft2(F_pre))#返回空间域
    f_pre=np.clip(f_pre,0,255).astype(np.uint8)#拉回0-255

    return g, f_pre


if __name__ == "__main__":
    # 读取图像并转换为灰度
    img = Image.open('test3.jpg')
    gray = rgb2gray(img)

    # 维纳滤波（三种K）
    tuihua,out=weina(gray, K=0.001)
    _,out2=weina(gray, K=0.01)
    _,out3=weina(gray, K=0.1)

    # 约束最小二乘方滤波（三种r）
    _,out4=min_ercheng(gray, r=0.001)
    _,out5=min_ercheng(gray, r=0.01)
    _,out6=min_ercheng(gray, r=0.1)

    plt.figure(figsize=(14, 10))

    #原图+退化图+维纳滤波
    plt.subplot(2, 4, 1)
    plt.imshow(gray, cmap='gray')
    plt.title("Original")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 4, 2)
    plt.imshow(tuihua, cmap='gray')
    plt.title("tuihua")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 4, 3)
    plt.imshow(out, cmap='gray')
    plt.title("weina K=0.001")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 4, 4)
    plt.imshow(out2, cmap='gray')
    plt.title("weina K=0.01")
    plt.xticks([])
    plt.yticks([])

    #维纳K=0.1+三种r
    plt.subplot(2, 4, 5)
    plt.imshow(out3, cmap='gray')
    plt.title("weina K=0.1")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 4, 6)
    plt.imshow(out4, cmap='gray')
    plt.title("min_ercheng r=0.001")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 4, 7)
    plt.imshow(out5, cmap='gray')
    plt.title("min_ercheng r=0.01")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 4, 8)
    plt.imshow(out6, cmap='gray')
    plt.title("min_ercheng r=0.1")
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()
    plt.show()
