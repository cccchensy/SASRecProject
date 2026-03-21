import os
import pickle
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# 导入项目中另外两个模块的类
from sasrec_dataset import SASRecDataset
from sasrec_model import SASRec

def evaluate(model, data_loader, item_emb_weight, device):
    """
    在验证集或测试集上评估模型性能
    计算 Hit Rate @ 10 和 NDCG @ 10
    """
    model.eval()
    HT = []
    NDCG = []
    
    with torch.no_grad():
        for user, seq_input, target_item, target_neg in data_loader:
            user = user.to(device)
            seq_input = seq_input.to(device)
            target_item = target_item.to(device)
            target_neg = target_neg.to(device)
            
            seq_out = model(seq_input)
            final_feat = seq_out[:, -1, :] 
            
            pos_emb = item_emb_weight(target_item) 
            neg_emb = item_emb_weight(target_neg) 
            
            pos_logits = (final_feat * pos_emb).sum(dim=-1, keepdim=True)
            neg_logits = torch.bmm(neg_emb, final_feat.unsqueeze(-1)).squeeze(-1)
            
            logits = torch.cat([pos_logits, neg_logits], dim=-1)
            _, indices = torch.sort(logits, dim=-1, descending=True)
            
            for i in range(indices.shape[0]):
                rank_list = indices[i][:10].tolist()
                if 0 in rank_list:
                    HT.append(1)
                    rank = rank_list.index(0)
                    NDCG.append(1.0 / np.log2(rank + 2))
                else:
                    HT.append(0)
                    NDCG.append(0)
                    
    return np.mean(HT), np.mean(NDCG)

def train_model(model, train_loader, val_loader, epochs, lr, device):
    """
    模型训练主循环
    """
    bce_criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    item_emb_weight = model.item_emb
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (user, seq_input, target_pos, target_neg) in enumerate(train_loader):
            seq_input = seq_input.to(device)
            target_pos = target_pos.to(device)
            target_neg = target_neg.to(device)
            
            optimizer.zero_grad()
            
            seq_out = model(seq_input)
            
            pos_emb = item_emb_weight(target_pos)
            neg_emb = item_emb_weight(target_neg)
            
            pos_logits = (seq_out * pos_emb).sum(dim=-1)
            neg_logits = (seq_out * neg_emb).sum(dim=-1)
            
            valid_mask = (target_pos != 0).float()
            
            pos_labels = torch.ones(pos_logits.shape, device=device)
            neg_labels = torch.zeros(neg_logits.shape, device=device)
            
            loss_pos = bce_criterion(pos_logits, pos_labels)
            loss_neg = bce_criterion(neg_logits, neg_labels)
            loss = (loss_pos + loss_neg) * valid_mask
            
            loss = loss.sum() / (valid_mask.sum() + 1e-8)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch:3d} | Train Loss: {total_loss / len(train_loader):.4f}")
        
        # 每隔 5 个 Epoch 评估一次，避免频繁评估拖慢训练速度
        if epoch % 5 == 0:
            hr, ndcg = evaluate(model, val_loader, item_emb_weight, device)
            print(f"--- Eval @ Epoch {epoch} | HR@10: {hr:.4f} | NDCG@10: {ndcg:.4f} ---")


if __name__ == '__main__':
    # 1. 设置超参数与运行设备
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的计算设备: {DEVICE}")
    
    DATA_DIR = './sasrec_features'
    MAX_SEQ_LEN = 50
    BATCH_SIZE = 256
    HIDDEN_UNITS = 50
    NUM_HEADS = 1
    NUM_BLOCKS = 2
    DROPOUT_RATE = 0.2
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    
    # 2. 加载序列数据与字典
    print("正在加载数据与字典...")
    seq_df = pd.read_parquet(os.path.join(DATA_DIR, 'sasrec_sequences.parquet'))
    # 将 DataFrame 转换为 user_id -> sequence_list 的字典形式
    user_seq_dict = dict(zip(seq_df['user_id'], seq_df['item_sequence']))
    
    with open(os.path.join(DATA_DIR, 'item2id.pkl'), 'rb') as f:
        item2id = pickle.load(f)
    ITEM_NUM = len(item2id)
    
    # 3. 实例化 Dataset 与 DataLoader
    print("正在构建数据加载器 (DataLoader)...")
    train_dataset = SASRecDataset(user_seq_dict, MAX_SEQ_LEN, ITEM_NUM, mode='train')
    val_dataset = SASRecDataset(user_seq_dict, MAX_SEQ_LEN, ITEM_NUM, mode='val')
    # 注意：在真实的 Windows 环境中，num_workers 建议先设置为 0 进行调试，稳定后再调大
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 4. 初始化 SASRec 模型
    print("正在初始化 SASRec 模型...")
    model = SASRec(
        item_num=ITEM_NUM,
        max_seq_len=MAX_SEQ_LEN,
        hidden_units=HIDDEN_UNITS,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        dropout_rate=DROPOUT_RATE,
        device=DEVICE
    ).to(DEVICE)
    
    # 5. 启动训练主循环
    print("开始训练...")
    train_model(model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, DEVICE)