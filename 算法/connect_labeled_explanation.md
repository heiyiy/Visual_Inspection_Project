# 连通成分标记（Connected Component Labeling）— 代码详解与优缺点分析

## 概述

**连通成分标记**是二值图像分析中的基础算法。它的任务很简单：在一幅二值图像中，把所有互相连通的前景像素（值为 1）归为同一个"连通区域"，并给每个区域分配一个唯一的数字标签。

```
输入二值图:              输出标记图:
1 1 0 0 0 1 1          1 1 0 0 0 2 2
1 1 0 0 0 0 1    →     1 1 0 0 0 0 2
0 0 0 1 1 0 0          0 0 0 3 3 0 0
0 1 0 1 1 0 0          0 4 0 3 3 0 0

标签 1: 左上角 2×2 的块      标签 3: 右下方的 2×2 块
标签 2: 右上角的 L 形块      标签 4: 左下角孤立点
```

### 什么是"连通"？

两个像素连通，意味着从一个像素出发，通过一系列相邻的像素，能走到另一个像素。根据邻域的定义分为两种：

| 类型 | 邻域 | 示意图 |
|------|------|------|
| **4 连通** | 上下左右 | `┼`（十字） |
| **8 连通** | 上下左右 + 四个对角 | `█`（3×3 方块除去中心） |

```
4连通邻域:              8连通邻域:
  □ □ □                  ■ ■ ■
  ■ ○ ■                  ■ ○ ■
  □ □ □                  ■ ■ ■
(□=不考虑, ■=邻居, ○=当前像素)
```

> 同一个二值图用 4 连通和 8 连通可能得到不同的标记结果。本代码两种都支持，通过 `connectivity` 参数切换。

---

## 完整代码逐行解释

### 第 1-4 行：导入依赖

```python
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from img_trans import rgb2gray
```

| 行号 | 代码 | 解释 |
|------|------|------|
| 1 | `import numpy as np` | 用于数组创建（`np.zeros`）、切片等操作。 |
| 2 | `from PIL import Image` | 读取图像文件。 |
| 3 | `import matplotlib.pyplot as plt` | 用于可视化输出——在标记图上叠加标签数字。 |
| 4 | `from img_trans import rgb2gray` | 复用项目已有的灰度转换函数。 |

---

### 第 7 行：函数定义

```python
def connect_label(img, connectivity=8):
```

| 参数 | 说明 |
|------|------|
| `img` | PIL 图像对象（彩色或灰度均可） |
| `connectivity` | `8`（默认）使用 8 连通；其他值使用 4 连通 |
| 返回值 | `h×w` 的 int 数组，0 表示背景，1,2,3… 表示各连通区域的标签 |

---

### 第 9-12 行：图像预处理（二值化）

```python
    img_gray = rgb2gray(img)
    img_array = np.array(img_gray)
    t = img_array < 200
    binary_img = t.astype(int)
```

| 行号 | 代码 | 解释 |
|------|------|------|
| 9 | `img_gray = rgb2gray(img)` | 调用项目的 `rgb2gray` 转为灰度图（若输入已经是灰度图也安全）。 |
| 10 | `img_array = np.array(img_gray)` | PIL 图像 → NumPy 数组，便于后续布尔索引。 |
| 11 | `t = img_array < 200` | **阈值二值化**。灰度值 < 200 的像素（暗色）判为前景（`True`），≥200 的判为背景（`False`）。这个阈值适合**白底黑字**的图像。 |
| 12 | `binary_img = t.astype(int)` | 布尔型 → int 型：`True→1`（前景）、`False→0`（背景）。 |

> **约定**：标记的是 **1（前景/黑色）的连通区域**。如果你的图像是黑底白字，需要反转阈值（`img_array > 200`）。

---

### 第 14-15 行：初始化

```python
    h, w = binary_img.shape
    output = np.zeros((h, w), dtype=int)
```

| 行号 | 代码 | 解释 |
|------|------|------|
| 14 | `h, w = binary_img.shape` | 获取图像尺寸。 |
| 15 | `output = np.zeros((h, w), dtype=int)` | 创建与输入同尺寸的全零数组，用于存放标签。0 表示未标记（初始值），正整数表示该像素所属的连通区域编号。`dtype=int` 避免默认 `float64`——标签是整数。 |

---

### 第 17 行：标签计数器

```python
    current_label = 1
```

标签从 1 开始。0 留给背景。每发现一个新的连通区域，`current_label` 加 1。

---

### 第 19-24 行：定义邻域偏移

```python
    if connectivity == 8:
        a = [(-1,-1), (-1,0), (-1,1),
             (0,-1),          (0,1),
             (1,-1),  (1,0),  (1,1)]
    else:
        a = [(0,-1),  (0,1),  (1,0),  (-1,0)]
```

| 行号 | 代码 | 解释 |
|------|------|------|
| 20-22 | 8 邻域偏移列表 | 包含周围 8 个方向（上、下、左、右 + 四个对角）。共 8 个 `(dx, dy)` 对。 |
| 24 | 4 邻域偏移列表 | 仅包含十字方向的 4 个邻居。共 4 个 `(dx, dy)` 对。 |

每个 `(dx, dy)` 是相对于当前像素的偏移量：`dx` 是行偏移（向下为正），`dy` 是列偏移（向右为正）。

---

### 第 26-40 行：核心算法 — DFS 洪水填充

这是整段代码的**核心**，使用了**深度优先搜索（DFS）**进行洪水填充（Flood Fill）。

```python
    for i in range(h):
        for j in range(w):
            if binary_img[i,j] == 1 and output[i,j] == 0:
                output[i,j] = current_label          # ① 标记起点
                stack = [(i,j)]                       # ② 起点入栈
                while stack:                          # ③ 栈非空就继续
                    (x,y) = stack.pop()               # ④ 弹出一个像素
                    for (dx,dy) in a:                 # ⑤ 遍历它的邻居
                        nx, ny = x+dx, y+dy           # ⑥ 邻居坐标
                        if 0 <= nx < h and 0 <= ny < w:     # ⑦ 边界检查
                            if binary_img[nx,ny] == 1 and output[nx,ny] == 0:  # ⑧ 是前景且未标记？
                                output[nx,ny] = current_label   # ⑨ 标记
                                stack.append((nx,ny))           # ⑩ 邻居入栈
                current_label = current_label + 1    # ⑪ 当前区域处理完毕
```

#### 逐步拆解

| 步骤 | 行号 | 代码 | 含义 |
|------|------|------|------|
| ① | 28-29 | `if binary_img[i,j]==1 and output[i,j]==0:` | 找到**未标记的前景像素**作为新区域的种子点。 |
| ② | 29 | `output[i,j] = current_label` | 给种子点打上当前标签。 |
| ③ | 30 | `stack = [(i,j)]` | 创建一个Python列表作为**栈**，种子点入栈。 |
| ④ | 31 | `while stack:` | DFS 循环：栈非空就一直处理。 |
| ⑤ | 32 | `(x,y) = stack.pop()` | **弹栈**（LIFO 后进先出），取出最近加入的像素。 |
| ⑥ | 33 | `for (dx,dy) in a:` | 遍历该像素的所有邻居方向。 |
| ⑦ | 34 | `nx, ny = x+dx, y+dy` | 计算邻居的绝对坐标。 |
| ⑧ | 35 | `0 <= nx < h and 0 <= ny < w` | 边界检查，防止坐标越界。 |
| ⑨ | 36 | `binary_img[nx,ny]==1 and output[nx,ny]==0` | 邻居是前景**且未被标记**。 |
| ⑩ | 37 | `output[nx,ny] = current_label` | 给邻居打上当前标签。 |
| ⑪ | 38 | `stack.append((nx,ny))` | 将该邻居压入栈，之后会处理它的邻居。 |
| ⑫ | 40 | `current_label += 1` | 当前连通区域的所有像素都被标记完毕（栈清空），标签号 +1 准备处理下一个区域。 |

#### DFS 过程动画示例

```
二值图 (8连通):
1 1 0
0 1 0
0 0 0

步骤1: 扫描到 (0,0)=1 → 标签=1, stack=[(0,0)]
步骤2: pop (0,0) → 邻居 (0,1) 和 (1,1) 是1 → 标记, stack=[(0,1),(1,1)]
步骤3: pop (1,1) → 无新邻居 → stack=[(0,1)]
步骤4: pop (0,1) → 无新邻居 → stack=[] → 结束

结果:
1 1 0
0 1 0
0 0 0
```

---

### 第 42-43 行：输出统计与返回

```python
    print(f"共有{current_label-1}个连通区域")
    return output
```

- `current_label - 1`：因为每次发现新区域后 +1，最终值比实际区域数多 1。
- `output`：大小为 `(h,w)` 的标签图。每个位置的值是该像素所属连通区域的编号（0=背景）。

---

### 第 45-62 行：可视化函数 `plot_results`

```python
def plot_results(output, save_path):
    rows, cols = output.shape
    plt.figure(figsize=(10, 5))
    plt.imshow(output, cmap='binary', interpolation='nearest')
    plt.axis('off')
    plt.grid(True, color='black', linewidth=0.5, linestyle='-', which='both')
    plt.xticks(np.arange(-0.5, cols, 1), minor=True)
    plt.yticks(np.arange(-0.5, rows, 1), minor=True)

    for i in range(rows):
        for j in range(cols):
            label_val = output[i, j]
            if label_val > 0:
                plt.text(j, i, str(label_val), color='white',
                         fontsize=10, ha='center', va='center',
                         fontweight='bold', zorder=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.show()
```

| 行号 | 代码 | 解释 |
|------|------|------|
| 47 | `rows, cols = output.shape` | 获取标记图的尺寸。 |
| 48 | `plt.figure(figsize=(10,5))` | 创建画布。 |
| 49 | `plt.imshow(..., cmap='binary')` | 以二值色彩映射显示（1=白色，0=黑色）。 |
| 49 | `interpolation='nearest'` | 最近邻插值——每个像素显示为清晰的方块，不做平滑，适合展示离散标签。 |
| 50 | `plt.axis('off')` | 隐去坐标轴。 |
| 51-53 | `plt.grid` + `xticks`/`yticks` | 在每个像素边界画黑色网格线，使每个像素的边界清晰可见。`np.arange(-0.5, cols, 1)` 将刻度线精确放置在像素交界处。 |
| 55-59 | `for i... for j...` | 双重循环遍历每个像素：若标签 > 0，在该像素中心用 `plt.text` 绘制白字标签号。 |
| 61 | `plt.savefig(save_path, dpi=100)` | 保存结果到文件。 |
| 62 | `plt.show()` | 弹出显示窗口。 |

---

### 第 64-67 行：主程序

```python
if __name__ == "__main__":
    image = Image.open("test.png")
    labeled_img = connect_label(image, connectivity=8)
    plot_results(labeled_img, save_path="connected_labels.png")
```

一行读图 → 一行标记 → 一行可视化，简洁清晰。

---

## 算法流程图

```
开始
  │
  ▼
扫描全图，找未标记的前景像素
  │
  ├── 找到？─── 是 ──→ 分配新标签，该像素入栈
  │                     │
  │                     ▼
  │                 栈为空？── 否 ──→ 弹出一个像素
  │                     │               │
  │                     │               ▼
  │                     │          遍历该像素的所有邻居
  │                     │               │
  │                     │               ▼
  │                     │          邻居是前景且未标记？
  │                     │           │         │
  │                     │         是         否
  │                     │           │         │
  │                     │           ▼         │
  │                     │      标记邻居       │
  │                     │      邻居入栈       │
  │                     │           │         │
  │                     │           └────┬────┘
  │                     │                │
  │                     │                ▼
  │                     │          继续循环 ←┘
  │                     │
  │                     ▼
  │                 栈为空 → 标签号+1 → 继续扫描
  │
  └── 没找到 ──→ 结束，输出标记图
```

---

## 优缺点分析

### 优点

| 优点 | 说明 |
|------|------|
| **实现简单直观** | DFS 洪水填充是连通标记最自然的方式，代码不到 40 行，逻辑清晰易懂。 |
| **一次扫描即可** | 与两遍扫描法（Two-Pass）不同，本算法无需处理等价标签表，直接 DFS 填满一个区域再继续。 |
| **支持 4/8 连通切换** | 通过 `a` 列表灵活切换邻域定义，适应不同场景需求。 |
| **适合教学** | 栈和 DFS 是计算机科学基础概念，这段代码是理解图遍历算法的绝佳实例。 |
| **无需额外依赖** | 仅用 NumPy 和 PIL，不依赖 OpenCV 等重型库。 |
| **输出直观** | `plot_results` 在每个连通区域上标注编号，方便验证和调试。 |

### 缺点

| 缺点 | 说明 | 影响 |
|------|------|------|
| **Python 递归/栈开销大** | 虽然用了显式栈避免递归深度限制，但 Python 的 `list.pop()` 和 `list.append()` 在像素量很大时效率不如 C 实现。 | 大图像（> 2000×2000）可能较慢。 |
| **内存占用** | `output` 和 `binary_img` 各占 `h×w×8` 字节（int64），加上 `stack` 最坏情况可容纳整个图像的所有前景像素。 | 对于千万像素级图像，内存压力明显。 |
| **非就地修改** | 每次都创建新的 `output` 数组，如果只需要覆盖原图则浪费了一半内存。 | 可通过 `in-place` 修改 `binary_img` 来优化。 |
| **阈值硬编码** | `t = img_array < 200` 的阈值写死在函数内，不适合所有图像。 | 对于光照不均匀的图像，固定阈值可能产生错误的前景/背景分割。 |
| **无区域属性输出** | 只输出了标签图，没有返回每个区域的面积、质心、外接矩形等信息。 | 实际应用通常需要这些属性做后续分析。 |
| **标签编号不稳定** | 标签号取决于扫描顺序（从左到右、从上到下），不同的扫描顺序会产生不同编号，缺乏语义意义。 | 对结果无实质影响，但不利于多图对比。 |

### 改进建议

1. **阈值参数化**：将 `threshold=200` 作为函数参数暴露
2. **返回区域属性**：遍历 `output` 统计每个标签的面积、质心、bbox
3. **大图优化**：使用 `collections.deque` 替代 `list` 做栈，或改用迭代 BFS（队列）
4. **彩色可视化**：用随机颜色渲染不同标签，视觉区分度更好
5. **等价标签表**：如果追求极致性能，可改为经典的两遍扫描法（Two-Pass），避免 DFS 的栈操作
