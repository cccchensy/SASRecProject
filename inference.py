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
    
    # 1. 加载模型训练时使用的基础字典 (Raw ID -> Model ID)
    dict_path = os.path.join(data_dir, 'item2id.pkl')
    if not os.path.exists(dict_path):
        raise FileNotFoundError(f"未找到字典文件: {dict_path}")
        
    with open(dict_path, 'rb') as f:
        item2id = pickle.load(f)
        
    # 2. 加载我们刚刚做好的中文翻译字典 (Raw ID -> 中文名)
    # 注意：确保这个 pkl 文件与你的 inference.py 在同一个目录下
    chinese_dict_path = 'id2name.pkl' 
    if os.path.exists(chinese_dict_path):
        with open(chinese_dict_path, 'rb') as f:
            rawid2chinese = pickle.load(f)
    else:
        print("[Warning] 未找到 id2name.pkl，将默认使用原始 ID 展示。")
        rawid2chinese = {}
        
    # 3. 核心修改：构建 Model ID 到 中文名 的直接映射
    # id2item 字典的 Key 是模型连续 ID，Value 直接变成了流畅的中文番剧名
    id2item = {}
    for raw_id, model_id in item2id.items():
        # raw_id 需要转为字符串以匹配我们刚刚生成的字典
        id2item[model_id] = rawid2chinese.get(str(raw_id), f"未知番剧_{raw_id}")
        
    item_num = len(item2id)
    
    # ... (下方初始化 SASRec 模型和加载权重的代码保持完全不变) ...
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
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    
    return model, item2id, id2item, max_seq_len, device

def interactive_recommendation(model, id2item, max_seq_len, device):
    """
    交互式冷启动推荐与 I2I 相似推荐逻辑 (加入 E&E 动态强化探测策略)
    """
    valid_ids = [item_id for item_id in id2item.keys() if item_id != 0]
    # 稍微扩大热门池以保证足够的随机探索空间
    popular_pool = sorted(valid_ids)[:500]
    
    print("\n" + "="*60)
    print("欢迎使用番剧推荐系统 - 动态强化冷启动推断")
    print("="*60)
    
    user_sequence = []
    seen_items = set()
    target_n = 5
    
    print(f"\n[Step 1] 构建初始兴趣序列 (目标: 至少收集 {target_n} 部)")
    print("输入说明: y(看过/喜欢), n(没看过/不喜欢), f(提前结束并推荐)\n")
    
    # E&E 策略计数器：控制智能预测与随机探索的比例
    smart_recommend_counter = 0
    
    while len(user_sequence) < target_n:
        # 策略分支 1: 序列为空，或达到了 3 次智能推荐的阈值，触发【随机探索】
        if not user_sequence or smart_recommend_counter >= 3:
            available_popular = list(set(popular_pool) - seen_items)
            if available_popular:
                item_id = random.choice(available_popular)
            else:
                available_global = list(set(valid_ids) - seen_items)
                if not available_global:
                    break
                item_id = random.choice(available_global)
            
            if user_sequence:
                tag = "[随机探索]"
                smart_recommend_counter = 0 # 触发随机后，重置计数器
            else:
                tag = "[热门初始]"
                
        # 策略分支 2: 已有 y 记录，进行实时张量推断，触发【智能关联】
        else:
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
                logits = torch.matmul(final_feat, all_item_emb.transpose(0, 1)).squeeze(0)
                
                # 负反馈屏蔽机制：将所有看过的 (y) 和 标记为不要的 (n) 全部屏蔽
                for seen_id in seen_items:
                    logits[seen_id] = -1e9
                logits[0] = -1e9 # 屏蔽 Padding
                
                # 选取当前序列下概率最高的那一部番剧
                item_id = torch.argmax(logits).item()
                
            tag = "[智能关联]"
            smart_recommend_counter += 1

        anime_name = id2item[item_id]
        ans = input(f"{tag} 《{anime_name}》: ").strip().lower()
        
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

    # 阶段 2：达标后的自由探索阶段，依然保持这种动态关联逻辑
    if len(user_sequence) >= target_n:
        print(f"\n[Info] 已达标 ({target_n}部)。您可以继续标记以提升模型精度，或输入 'f' 立即获取最终推荐。")
        while True:
            if smart_recommend_counter >= 3:
                available_ids = list(set(popular_pool) - seen_items)
                if not available_ids:
                    available_ids = list(set(valid_ids) - seen_items)
                if not available_ids:
                    break
                item_id = random.choice(available_ids)
                tag = "[随机探索]"
                smart_recommend_counter = 0
            else:
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
                    logits = torch.matmul(final_feat, model.item_emb.weight.transpose(0, 1)).squeeze(0)
                    for seen_id in seen_items:
                        logits[seen_id] = -1e9
                    logits[0] = -1e9 
                    item_id = torch.argmax(logits).item()
                tag = "[智能关联]"
                smart_recommend_counter += 1
                
            anime_name = id2item[item_id]
            ans = input(f"{tag} 《{anime_name}》: ").strip().lower()
            
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

    # ==========================================
    # 以下为生成最终推荐的代码，保持完全不变
    # ==========================================
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
        
    print("\n" + "="*60)
    print(" [Step 3] 看了又看：深度关联探索 (基于 Top-3 推荐)")
    print("="*60)
    
    top3_indices = top_indices[:3]
    norm_item_emb = F.normalize(all_item_emb, p=2, dim=1)
    
    with torch.no_grad():
        for rank, target_item_id in enumerate(top3_indices, 1):
            target_name = id2item.get(target_item_id, "未知番剧")
            print(f"\n >>> 因为为您推荐了: 《{target_name}》")
            print("     看这部番的用户，强烈关联了以下作品：")
            
            target_vec = norm_item_emb[target_item_id].unsqueeze(0)
            sim_scores = torch.matmul(target_vec, norm_item_emb.transpose(0, 1)).squeeze(0)
            
            sim_scores[target_item_id] = -1.0
            sim_scores[0] = -1.0
            for seen_id in seen_items:
                sim_scores[seen_id] = -1.0
                
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