import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from img_trans import rgb2gray


def histogram_equalization(img):
    '''
    对灰度图像进行直方图均衡化
    '''
    gray_img=rgb2gray(img)
    #计算灰度图像的直方图,flatten()将二维图像展平为一维数组，bins=256表示灰度级数为256，
    # range=(0,255)表示灰度值的范围,返回值hist是一个长度为256的一维数组，表示每个灰度级别的像素数量
    #_丢弃bin的边界值
    sk,_=np.histogram(gray_img.flatten(),bins=256,range=(0,256))
    sk_sum=sk.sum()
    sk1=sk/sk_sum#计算每个灰度级别的频次，每个灰度级别的像素数量除以总像素数量
    tk=np.cumsum(sk1)#计算累积分布函数（CDF），即频次的累积和，表示小于或等于每个灰度级别的像素比例
    tk=np.round(tk*255).astype(np.uint8)#四舍五入
    equalized_img=tk[gray_img]#映射，gray_img中的每个值被用作tk数组的索引
    
    equalized_hist,_=np.histogram(equalized_img.flatten(),bins=256,range=(0,256))
    
    return gray_img,equalized_img,sk,equalized_hist

def target_cdf(gray_levels):#定义目标累计cdf函数
    # gray_levels=255-gray_levels
    exp_pmf_unnormalized = np.exp(-0.02 * gray_levels) #指数递减
    pmf = exp_pmf_unnormalized / np.sum(exp_pmf_unnormalized)
    target = np.cumsum(pmf)
    
    return target/target[-1]

def histogram_standardization(img,mode="gml"):
    '''
    对灰度图像进行直方图规定化
    '''
    gray_img=rgb2gray(img)
    sk,_=np.histogram(gray_img.flatten(),bins=256,range=(0,256))
    sk1=sk/sk.sum()
    tk=np.cumsum(sk1)#累计频次
    
    all_gray_levels=np.arange(256)#创建0-255的一维数组,对应线性cdf
    target=target_cdf(all_gray_levels)
    
    standardized_hist=np.zeros(256,dtype=float)
    #SML映射
    if mode=='sml':
        a_to_b=np.zeros(256,dtype=int)#院士到目标的映射
        for i in range(256):
            diff=np.abs(tk[i]-target)
            b=np.argmin(diff)#argmin 遇到相同最小值时默认返回第一个
            a_to_b[i]=b
            
        for j in range(256):
            b=a_to_b[j]
            standardized_hist[b]+=sk[j]#当有多个院士灰度级对应同一个目标灰度级时，累加成该目标灰度级最后的概率。没有院士对应的目标概率为0
        
        standardized_img=a_to_b[gray_img]
        
    #GML映射
    if mode=="gml":
        #先找到不重复的目标灰度级和对应cdf
        t=-1.0
        target_val=[]
        for k in range(256):
            current=target[k]
            if current!=t:
                target_val.append((k,current))#填入（灰度级，概率）数组
                t=current
        a_to_b=np.zeros(256,dtype=int)
        
        for i in range(256):#遍历原始
            b=-1
            min_diff=1.1
            #遍历目标，找最近的
            for b,b_val in target_val:
                diff=np.abs(tk[i]-b_val)
                if diff<min_diff:
                    min_diff=diff
                    b_final=b
            a_to_b[i]=b_final
            standardized_hist[b_final]+=sk[i]#当有多个院士灰度级对应同一个目标灰度级时，累加成该目标灰度级最后的概率。没有院士对应的目标概率为0
        
        standardized_hist*=gray_img.size
        standardized_img=a_to_b[gray_img]
    
    
    return gray_img,standardized_img,sk,standardized_hist

def plot_histogram_and_cdf(original,equalized,original_hist,result_hist):
    """展示原图、处理后图像及其直方图"""
    
    fig,axes=plt.subplots(2,2,figsize=(10,10))
    
    #原图
    axes[0,0].imshow(original,cmap='gray')
    axes[0,0].set_title("Original Image")
    axes[0,0].axis('off')
    
    #处理后图像
    axes[0,1].imshow(equalized,cmap='gray')
    axes[0,1].set_title("Equalized Image")
    axes[0,1].axis('off')
    
    #原图直方图
    axes[1,0].plot(original_hist,color='blue')
    axes[1,0].set_title("Original Histogram")
    axes[1,0].set_xlabel("Pixel Intensity")
    axes[1,0].set_ylabel("number of pixels")
    
    #处理后图像实际直方图
    axes[1,1].plot(result_hist,color='blue')
    axes[1,1].set_title("Equalized Histogram(lilun)")
    axes[1,1].set_xlabel("Pixel Intensity")
    axes[1,1].set_ylabel("number of pixels")
    
    plt.tight_layout()
    plt.show()
    
if __name__=="__main__":
    img=Image.open("test2.jpeg")
    original_img,equalized_img,original_hist,equlized_hist=histogram_equalization(img)
    plot_histogram_and_cdf(original_img,equalized_img,original_hist,equlized_hist)
    plt.savefig("histogram_result.png")