# 交互式番剧推荐系统 (SASRec-Based Anime Recommender)

本项目是一个基于 SASRec (Self-Attentive Sequential Recommendation) 架构的深度学习序列推荐系统。
模型已在 2.2 亿条真实用户交互数据上完成预训练，能够精准捕捉番剧观看的上下文依赖与长短期兴趣。

推断模块集成了 E&E动态探测策略与基于高维向量的 I2I 关联推荐

## 1. 目录结构

```text
SASRecProject/
├── sasrec_model.py          # 模型神经网络架构定义
├── recommender_engine.py     # 推荐引擎核心（session 管理 + E&E 策略 + 推荐逻辑）
├── inference.py              # 命令行交互式推断入口
├── web_app.py                # FastAPI Web 后端
├── requirements.txt          # Python 依赖
├── id2name.pkl               # 番剧 ID 到中文译名的映射字典
├── sasrec_features/
│   └── item2id.pkl           # 基础特征索引字典
├── saved_models/
│   └── sasrec_model_best.pth # 预训练最佳模型权重
└── static/
    ├── index.html            # 前端页面
    ├── style.css             # 样式
    └── main.js               # 前端逻辑
```

## 2. 环境部署

推荐使用 Anaconda 构建虚拟环境。请在终端执行以下命令：

1. 创建并激活虚拟环境 (推荐 Python 3.9+)(不建议使用 Python 3.12+)：
- conda create -n anime_rec python=3.10 -y
- conda activate anime_rec

2. 安装核心依赖：
- pip install -r requirements.txt

注：本推断程序原生支持 CPU 运行。如果您的设备支持 CUDA，程序将自动调用 GPU 以加速张量乘法运算。

## 3. 命令行运行方式

在项目根目录下，直接运行推断脚本：

```bash
python inference.py
```

## 4. 交互模式说明

程序启动后，将进入冷启动兴趣收集阶段（目标收集 5 部感兴趣的番剧）。系统将使用以下两种策略向您提问：

- **[热门初始] / [随机探索]**：从大众热门池中进行随机抽样，探索您的潜在兴趣边界，打破信息茧房。
- **[智能关联]**：当您标记了感兴趣的番剧后，模型将实时抓取您当前的兴趣序列，通过 50 维 Embedding 向量的自注意力机制，计算出下一步概率最高的番剧进行精准探测。

**输入指令说明：**
- `y` 或 `1` : 看过 / 感兴趣
- `n` 或 `0` : 没看过 / 不感兴趣 (该番剧及其底层权重将在后续推断中被降级)
- `f` : 提前结束收集，立即生成最终的 Top-10 推荐榜单及深度关联推断 (I2I)。

## 5. Web Demo 运行方式

除了命令行模式，本项目还提供了浏览器可访问的 Web Demo。

### 5.1 安装依赖

```bash
pip install -r requirements.txt
```

### 5.2 启动 Web 服务

在项目根目录下运行：

```bash
uvicorn web_app:app --reload
```

如需指定端口：

```bash
uvicorn web_app:app --reload --port 8001
```

### 5.3 浏览器访问

启动后打开浏览器访问：

```text
http://127.0.0.1:8000
```

### 5.4 Web Demo 功能

- 页面加载后自动显示第一部候选番剧（热门初始策略）
- 点击"看过 / 喜欢"或"没看过 / 不感兴趣"提交反馈
- 连续智能关联 3 次后自动切换为随机探索
- 点击"生成推荐"获取 Top-10 个性化推荐和看了又看 (I2I) 关联推荐
- 点击"重新开始"清空结果并创建新会话

### 5.5 注意事项

- 需要保证 `id2name.pkl` 存在于项目根目录
- 需要保证 `sasrec_features/item2id.pkl` 存在
- 需要保证 `saved_models/sasrec_model_best.pth` 存在
- 如果有 CUDA 会自动使用 GPU，否则使用 CPU
- 模型在服务启动时只加载一次，API 请求不会重复加载
