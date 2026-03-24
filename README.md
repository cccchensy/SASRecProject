# 交互式番剧推荐系统 (SASRec-Based Anime Recommender)

本项目是一个基于 SASRec (Self-Attentive Sequential Recommendation) 架构的深度学习序列推荐系统。
模型已在 2.2 亿条真实用户交互数据上完成预训练，能够精准捕捉番剧观看的上下文依赖与长短期兴趣。

推断模块集成了 E&E动态探测策略与基于高维向量的 I2I 关联推荐

## 1. 目录结构

确保您的运行目录包含以下核心文件：
- `inference.py` : 交互式推断主程序入口
- `sasrec_model.py` : 模型神经网络架构定义
- `id2name.pkl` : 番剧 ID 到中文译名的映射字典
- `sasrec_features/item2id.pkl` : 基础特征索引字典
- `saved_models/sasrec_model_best.pth` : 预训练最佳模型权重

## 2. 环境部署

推荐使用 Anaconda 构建虚拟环境。请在终端执行以下命令：

1. 创建并激活虚拟环境 (推荐 Python 3.9+)(不建议使用 Python 3.12+)：
conda create -n anime_rec python=3.10 -y
conda activate anime_rec

2. 安装核心依赖：
pip install -r requirements.txt

注：本推断程序原生支持 CPU 运行。如果您的设备支持 CUDA，程序将自动调用 GPU 以加速张量乘法运算。

## 3. 运行指南

在项目根目录下，直接运行推断脚本：

python inference.py

## 4. 交互模式说明

程序启动后，将进入冷启动兴趣收集阶段（目标收集 5 部感兴趣的番剧）。系统将使用以下两种策略向您提问：

- **[热门初始] / [随机探索]**：从大众热门池中进行随机抽样，探索您的潜在兴趣边界，打破信息茧房。
- **[智能关联]**：当您标记了感兴趣的番剧后，模型将实时抓取您当前的兴趣序列，通过 50 维 Embedding 向量的自注意力机制，计算出下一步概率最高的番剧进行精准探测。

**输入指令说明：**
- `y` 或 `1` : 看过 / 感兴趣
- `n` 或 `0` : 没看过 / 不感兴趣 (该番剧及其底层权重将在后续推断中被降级)
- `f` : 提前结束收集，立即生成最终的 Top-10 推荐榜单及深度关联推断 (I2I)。