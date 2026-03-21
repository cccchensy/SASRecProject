import torch
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np

# 1. 加载一个预训练模型 (比如 ResNet50)
model = models.resnet50(pretrained=True)

# 2. 提取某一层卷积核的权重 (比如 layer2 的第一个卷积层)
# 权重的形状通常是 [out_channels, in_channels, k, k]
weight = model.layer2[0].conv1.weight.detach().numpy()

# 3. 将其展开成 2D 矩阵 (这是 RMT 分析的标准做法)
# 将 [N, M, k, k] -> [N, M*k*k]
W = weight.reshape(weight.shape[0], -1)

# 4. 计算奇异值 (Singular Values)
# 只有对于方阵才是特征值，对于长方形矩阵我们看奇异值
u, s, vh = np.linalg.svd(W, full_matrices=False)

# 5. 画出奇异值的对数分布 (Log-Log Plot)
# 如果是分形/重尾的，这应该近似一条直线
plt.figure(figsize=(10, 6))
plt.hist(s, bins=100, density=True, log=True)
plt.title("Singular Value Distribution (Log Scale)")
plt.xlabel("Singular Value")
plt.ylabel("Density")
plt.show()