import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from img_trans import rgb2gray

##对二值化图像进行连通标记
def connect_label(img,connectivity=8):
    
    img_gray=rgb2gray(img)#转灰度
    img_array=np.array(img_gray)#转数组
    t=img_array<200#以200为阈值二值化，黑为1白为0
    binary_img=t.astype(int)#astype（int）转换成int型

    h,w=binary_img.shape
    output=np.zeros((h,w),dtype=int)#创建一个行r✖列c（高乘以宽）的空数组，用于后续填入，dtype要声明，不然默认是float型

    current_label=1#初始标签从一开始
    
    if connectivity==8:##8连通
       a=[(-1,-1),(-1,0),(-1,1),
          (0,-1),        (0,1),
          (1,-1), (1,0), (1,1)]
    else:##4连通
        a=[(-1,0),(0,-1),(0,1),(1,0)]
        
    for i in range(h):
        for j in range(w):
            if binary_img[i,j]==1 and output[i,j]==0:
                output[i,j]=current_label
                stack=[(i,j)]#定义一个栈来存储需要处理的像素点
                while stack:   #深度优先搜索（dfs），直到栈里的像素为空
                    (x,y)=stack.pop()#从栈里取出一个像素
                    for (dx,dy) in a:#遍历邻域
                        nx,ny=x+dx,y+dy#计算该点（x，y）的相邻点坐标
                        if 0<=nx<h and 0<=ny<w:#排查是否超出边界
                            if binary_img[nx,ny]==1 and output[nx,ny]==0:
                                output[nx,ny]=current_label#如果相邻点也是1且未标记，则标记为当前标签
                                stack.append((nx,ny))#将相邻点加入栈中，继续处理,之道stack为空，跳出循环
                                
                current_label=current_label+1#处理完一个连通区域后，标签加一，继续处理下一个区域
                
    print(f"共有{current_label-1}个连通区域")
    return output

def plot_results(output,save_path):

    rows,cols=output.shape
    plt.figure(figsize=(10,5))
    plt.imshow(output,cmap='binary',interpolation='nearest')
    plt.axis('off')#去掉坐标轴
    plt.grid(True, color='black', linewidth=0.5,linestyle='-',which='both')
    plt.xticks(np.arange(-0.5, cols, 1), minor=True)
    plt.yticks(np.arange(-0.5, rows, 1), minor=True)

    for i in range(rows):
        for j in range(cols):
            label_val=output[i,j]
            if label_val>0:
                plt.text(j,i,str(label_val),color='white',fontsize=10,ha='center',va='center',fontweight='bold',zorder=10)
    plt.tight_layout()
    plt.savefig(save_path,dpi=100)
    plt.show()

if __name__=="__main__":
    image=Image.open("test.png")
    labeled_img=connect_label(image,connectivity=8)#连通标记 
    plot_results(labeled_img,save_path="connected_labels.png")#绘制连通标记结果
    
    


             
                
                
    