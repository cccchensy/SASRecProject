import os
import torch
import pickle
import random
import numpy as np

# 导入你之前写好的模型结构
from sasrec_model import SASRec

def load_environment():
    """
    加载环境配置、字典与模型权重
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 推断程序已启动，当前计算设备: {device}")
    
    data_dir = './sasrec_features'
    model_dir = './saved_models'
    
    # 加载字典文件
    dict_path = os.path.join(data_dir, 'item2id.pkl')
    if not os.path.exists(dict_path):
        raise FileNotFoundError(f"未找到字典文件: {dict_path}")
        
    with open(dict_path, 'rb') as f:
        item2id = pickle.load(f)
        
    # 构建反向字典：通过 ID 查找番剧名称
    id2item = {v: k for k, v in item2id.items()}
    item_num = len(item2id)
    
    # 初始化模型 (超参数必须与训练时完全一致)
    max_seq_len = 50
    hidden_units = 50
    num_heads = 1
    num_blocks = 2
    
    model = SASRec(
        item_num=item_num,
        max_seq_len=max_seq_len,
        hidden_units=hidden_units,
        num_heads=num_heads,
        num_blocks=num_blocks,
        dropout_rate=0.0, # 推断时不需要 dropout
        device=device
    ).to(device)
    
    # 加载最优权重文件
    weight_path = os.path.join(model_dir, 'sasrec_model_best.pth')
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"未找到权重文件: {weight_path}，请确认模型已完成训练并保存。")
        
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval() # 切换到评估模式，关闭 Dropout 和 BatchNorm 的动态更新
    
    return model, item2id, id2item, max_seq_len, device

def interactive_recommendation(model, id2item, max_seq_len, device):
    """
    交互式冷启动推荐逻辑
    """
    # 提取所有合法的番剧 ID (排除作为 Padding 的 0)
    valid_ids = [item_id for item_id in id2item.keys() if item_id != 0]
    
    print("\n" + "="*50)
    print("欢迎使用番剧推荐系统 - 冷启动兴趣探测")
    print("="*50)
    
    # 随机抽取 10 部番剧
    sampled_ids = random.sample(valid_ids, 10)
    
    user_sequence = []
    seen_items = set()
    
    print("\n请根据以下番剧，输入 y(感兴趣) 或 n(不感兴趣/没看过)：")
    for item_id in sampled_ids:
        anime_name = id2item[item_id]
        while True:
            ans = input(f"《{anime_name}》: ").strip().lower()
            if ans in ['y', 'yes', '1']:
                user_sequence.append(item_id)
                seen_items.add(item_id)
                break
            elif ans in ['n', 'no', '0']:
                seen_items.add(item_id)
                break
            else:
                print("输入无效，请输入 y 或 n。")

    if not user_sequence:
        print("\n您没有选择任何感兴趣的番剧，系统无法进行有效推荐。程序结束。")
        return

    print("\n[*] 正在提取您的兴趣特征，生成专属推荐列表...\n")
    
    # 将用户的历史序列格式化为模型输入 (定长截断与左侧补零)
    seq_input = np.zeros(max_seq_len, dtype=np.int64)
    seq_len = len(user_sequence)
    
    if seq_len >= max_seq_len:
        seq_input[:] = user_sequence[-max_seq_len:]
    else:
        seq_input[-seq_len:] = user_sequence
        
    # 转换为 Tensor 格式
    seq_tensor = torch.tensor([seq_input], dtype=torch.long).to(device)
    
    with torch.no_grad():
        # 获取序列的隐层表征
        seq_out = model(seq_tensor) # shape: (1, max_seq_len, hidden_units)
        
        # 提取最后一个时间步的向量作为用户当前的动态兴趣画像
        final_feat = seq_out[:, -1, :] # shape: (1, hidden_units)
        
        # 获取整个番剧库的 Embedding
        # shape: (item_num + 1, hidden_units)
        all_item_emb = model.item_emb.weight 
        
        # 计算该用户兴趣向量与所有番剧向量的点积得分
        # shape: (1, item_num + 1)
        logits = torch.matmul(final_feat, all_item_emb.transpose(0, 1))
        
        # 将已经展示过或选择过的番剧得分降至极低，防止重复推荐
        for item_id in seen_items:
            logits[0, item_id] = -1e9
        logits[0, 0] = -1e9 # 排除 Padding ID
        
        # 获取得分最高的前 10 个番剧索引
        _, top_indices = torch.topk(logits[0], k=10)
        top_indices = top_indices.cpu().numpy().tolist()
        
    print("="*50)
    print("基于您的初始选择，我们为您推荐以下番剧：")
    print("="*50)
    for rank, item_id in enumerate(top_indices, 1):
        anime_name = id2item.get(item_id, "未知番剧")
        print(f"Top {rank}: 《{anime_name}》")
    print("="*50 + "\n")

if __name__ == '__main__':
    try:
        model_instance, item2id_dict, id2item_dict, seq_len_limit, current_device = load_environment()
        interactive_recommendation(model_instance, id2item_dict, seq_len_limit, current_device)
    except Exception as e:
        print(f"程序运行出错: {e}")