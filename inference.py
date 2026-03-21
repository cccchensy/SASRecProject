import os
import torch
import pickle
import random
import numpy as np
import torch.nn.functional as F

from sasrec_model import SASRec

def load_environment():
    """
    加载环境配置、字典与模型权重
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[System] 推断程序已启动，当前计算设备: {device}")
    
    data_dir = './sasrec_features'
    model_dir = './saved_models'
    
    dict_path = os.path.join(data_dir, 'item2id.pkl')
    if not os.path.exists(dict_path):
        raise FileNotFoundError(f"未找到字典文件: {dict_path}")
        
    with open(dict_path, 'rb') as f:
        item2id = pickle.load(f)
        
    id2item = {v: k for k, v in item2id.items()}
    item_num = len(item2id)
    
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
        dropout_rate=0.0,
        device=device
    ).to(device)
    
    weight_path = os.path.join(model_dir, 'sasrec_model_best.pth')
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"未找到权重文件: {weight_path}")
        
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    
    return model, item2id, id2item, max_seq_len, device

def interactive_recommendation(model, id2item, max_seq_len, device):
    """
    交互式冷启动推荐与 I2I 相似推荐逻辑
    """
    valid_ids = [item_id for item_id in id2item.keys() if item_id != 0]
    popular_pool = sorted(valid_ids)[:200]
    
    print("\n" + "="*60)
    print("欢迎使用番剧推荐系统 - 冷启动与深度关联推断")
    print("="*60)
    
    user_sequence = []
    seen_items = set()
    target_n = 5
    
    print(f"\n[Step 1] 构建初始兴趣序列 (目标: 至少收集 {target_n} 部)")
    print("输入说明: y(看过/喜欢), n(没看过/不喜欢), f(提前结束并推荐)\n")
    
    while len(user_sequence) < target_n:
        available_popular = list(set(popular_pool) - seen_items)
        
        if not available_popular:
            available_global = list(set(valid_ids) - seen_items)
            if not available_global:
                break
            item_id = random.choice(available_global)
        else:
            item_id = random.choice(available_popular)
            
        anime_name = id2item[item_id]
        ans = input(f"《{anime_name}》: ").strip().lower()
        
        if ans in ['f', 'finish', 'q', 'quit']:
            if len(user_sequence) < target_n:
                confirm = input(f"  -> 仅收集 {len(user_sequence)} 部可能导致推荐偏差，确认直接推荐？(y/n): ").strip().lower()
                if confirm in ['y', 'yes']:
                    break
                else:
                    continue
            else:
                break
        elif ans in ['y', 'yes', '1']:
            user_sequence.append(item_id)
            seen_items.add(item_id)
            print(f"  -> [记录成功] 进度: {len(user_sequence)} / {target_n}")
        elif ans in ['n', 'no', '0', '']:
            seen_items.add(item_id)
        else:
            print("  -> 输入无效，请输入 y, n, 或 f。")

    if len(user_sequence) >= target_n:
        print(f"\n[Info] 已达标 ({target_n}部)。可继续标记以提升精度，或输入 'f' 获取推荐。")
        while True:
            available_ids = list(set(valid_ids) - seen_items)
            if not available_ids:
                break
            
            item_id = random.choice(available_ids)
            anime_name = id2item[item_id]
            ans = input(f"《{anime_name}》: ").strip().lower()
            
            if ans in ['f', 'finish', 'q', 'quit', '']:
                break
            elif ans in ['y', 'yes', '1']:
                user_sequence.append(item_id)
                seen_items.add(item_id)
                print(f"  -> [记录成功] 总数: {len(user_sequence)}")
            elif ans in ['n', 'no', '0']:
                seen_items.add(item_id)

    if not user_sequence:
        print("\n[Error] 未提供有效序列，程序终止。")
        return

    print(f"\n[System] 正在基于 {len(user_sequence)} 部番剧进行张量推断...")
    
    seq_input = np.zeros(max_seq_len, dtype=np.int64)
    seq_len = len(user_sequence)
    
    if seq_len >= max_seq_len:
        seq_input[:] = user_sequence[-max_seq_len:]
    else:
        seq_input[-seq_len:] = user_sequence
        
    seq_tensor = torch.tensor([seq_input], dtype=torch.long).to(device)
    
    with torch.no_grad():
        seq_out = model(seq_tensor)
        final_feat = seq_out[:, -1, :] 
        all_item_emb = model.item_emb.weight 
        logits = torch.matmul(final_feat, all_item_emb.transpose(0, 1))
        
        for item_id in seen_items:
            logits[0, item_id] = -1e9
        logits[0, 0] = -1e9 
        
        _, top_indices = torch.topk(logits[0], k=10)
        top_indices = top_indices.cpu().numpy().tolist()
        
    print("\n" + "="*60)
    print(" [Step 2] 您的专属 Top-10 个性化推荐")
    print("="*60)
    for rank, item_id in enumerate(top_indices, 1):
        anime_name = id2item.get(item_id, "未知番剧")
        print(f" Top {rank:2d} | 《{anime_name}》")
        
    # ---------- 新增功能：基于 Top-3 的关联推荐 (I2I) ----------
    print("\n" + "="*60)
    print(" [Step 3] 看了又看：深度关联探索 (基于 Top-3 推荐)")
    print("="*60)
    
    top3_indices = top_indices[:3]
    
    # 对整个番剧 Embedding 矩阵进行 L2 归一化，以便计算余弦相似度
    norm_item_emb = F.normalize(all_item_emb, p=2, dim=1)
    
    with torch.no_grad():
        for rank, target_item_id in enumerate(top3_indices, 1):
            target_name = id2item.get(target_item_id, "未知番剧")
            print(f"\n >>> 因为为您推荐了: 《{target_name}》")
            print("     看这部番的用户，强烈关联了以下作品：")
            
            # 获取目标番剧的归一化向量 (1, hidden_units)
            target_vec = norm_item_emb[target_item_id].unsqueeze(0)
            
            # 计算目标向量与所有库内向量的余弦相似度
            sim_scores = torch.matmul(target_vec, norm_item_emb.transpose(0, 1)).squeeze(0)
            
            # 屏蔽目标番剧本身、Padding ID 以及用户已经看过的番剧
            sim_scores[target_item_id] = -1.0
            sim_scores[0] = -1.0
            for seen_id in seen_items:
                sim_scores[seen_id] = -1.0
                
            # 提取相似度最高的 Top-5
            _, sim_top_indices = torch.topk(sim_scores, k=5)
            sim_top_indices = sim_top_indices.cpu().numpy().tolist()
            
            for sim_rank, sim_item_id in enumerate(sim_top_indices, 1):
                sim_name = id2item.get(sim_item_id, "未知番剧")
                print(f"      - 《{sim_name}》")
                
    print("\n" + "="*60 + "\n")

if __name__ == '__main__':
    try:
        model_instance, item2id_dict, id2item_dict, seq_len_limit, current_device = load_environment()
        interactive_recommendation(model_instance, id2item_dict, seq_len_limit, current_device)
    except Exception as e:
        print(f"[Error] 程序运行异常: {e}")