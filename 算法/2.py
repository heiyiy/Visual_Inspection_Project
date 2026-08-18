import numpy as np

# 原始矩阵
img = np.array([
    [14, 11, 21, 16, 20],
    [24, 32, 64, 32, 24],
    [25, 26, 40, 32, 30],
    [32, 30, 33, 30, 32],
    [11, 33, 22, 11, 22]
], dtype=np.float32)


# =========================
# 局部增强（按你PDF的公式）
# =========================
def local_enhancement_pdf(image, k=0.6):
    h, w = image.shape
    
    # 全局均值 M
    M = np.mean(image)

    output = np.zeros((h-2, w-2))

    for i in range(1, h-1):
        for j in range(1, w-1):
            window = image[i-1:i+2, j-1:j+2]

            m_local = np.mean(window)
            sigma_local = np.std(window)

            # 防止除0
            if sigma_local == 0:
                A = 0
            else:
                A = k * M / sigma_local

            g = A * (image[i, j] - m_local) + m_local

            output[i-1, j-1] = g

    return np.round(output, 2)  # 保留小数方便看


# 计算
result = local_enhancement_pdf(img)

print("增强结果：")

print(result)