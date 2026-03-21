import torch
from torch.utils.data import Dataset
import numpy as np
import random

class SASRecDataset(Dataset):
    def __init__(self, user_seq_dict, max_len, item_num, mode='train'):
        """
        SASRec 数据集构建
        
        参数:
            user_seq_dict: 字典，键为 user_id，值为按时间排序的 item_id 列表
            max_len: 序列的最大截断长度 L
            item_num: 物品的总数 (用于负采样，物品 ID 范围应为 1 到 item_num)
            mode: 'train', 'val', 或 'test'
        """
        self.user_ids = list(user_seq_dict.keys())
        self.user_seq_dict = user_seq_dict
        self.max_len = max_len
        self.item_num = item_num
        self.mode = mode

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user = self.user_ids[idx]
        seq = self.user_seq_dict[user]
        
        # 1. 留一法 (Leave-one-out) 数据集划分
        # 假设用户序列为 [1, 2, 3, 4, 5]
        if self.mode == 'train':
            # 训练集使用除了最后两个物品之外的所有物品
            # 序列变为 [1, 2, 3]
            current_seq = seq[:-2]
        elif self.mode == 'val':
            # 验证集使用除了最后一个物品之外的所有物品
            # 序列变为 [1, 2, 3, 4]
            current_seq = seq[:-1]
        elif self.mode == 'test':
            # 测试集使用所有物品
            # 序列变为 [1, 2, 3, 4, 5]
            current_seq = seq
        else:
            raise ValueError("mode 必须是 'train', 'val', 或 'test'")

        # 将用户的全部历史转换为 set，用于后续快速判断负样本
        user_history_set = set(seq)

        # 2. 训练模式：构建 Input, Positive Target 和 Negative Target
        if self.mode == 'train':
            # 输入序列 (除去最后一个元素)
            seq_input = current_seq[:-1]
            # 正样本序列 (序列整体左移一位，即除去第一个元素)
            seq_target_pos = current_seq[1:]
            
            # 为每一个正样本，随机生成一个用户从未看过的负样本
            seq_target_neg = []
            for _ in seq_target_pos:
                neg_item = random.randint(1, self.item_num)
                while neg_item in user_history_set:
                    neg_item = random.randint(1, self.item_num)
                seq_target_neg.append(neg_item)
                
            # 执行截断与左侧补零 (Padding)
            tokens_input = np.zeros(self.max_len, dtype=np.int64)
            tokens_pos = np.zeros(self.max_len, dtype=np.int64)
            tokens_neg = np.zeros(self.max_len, dtype=np.int64)
            
            seq_len = len(seq_input)
            if seq_len > 0:
                if seq_len >= self.max_len:
                    tokens_input[:] = seq_input[-self.max_len:]
                    tokens_pos[:] = seq_target_pos[-self.max_len:]
                    tokens_neg[:] = seq_target_neg[-self.max_len:]
                else:
                    tokens_input[-seq_len:] = seq_input
                    tokens_pos[-seq_len:] = seq_target_pos
                    tokens_neg[-seq_len:] = seq_target_neg
                    
            return torch.tensor(user, dtype=torch.long), \
                   torch.tensor(tokens_input, dtype=torch.long), \
                   torch.tensor(tokens_pos, dtype=torch.long), \
                   torch.tensor(tokens_neg, dtype=torch.long)

        # 3. 验证/测试模式：构建 Input 和用于排序的 100 个候选物品 (1正 + 99负)
        else:
            # 验证/测试的输入序列是不包含当前目标物品的前置历史
            seq_input = current_seq[:-1]
            # 验证/测试的真实目标是当前序列的最后一个物品
            target_item = current_seq[-1]
            
            # 生成 99 个负样本
            negative_samples = []
            for _ in range(99):
                neg_item = random.randint(1, self.item_num)
                while neg_item in user_history_set:
                    neg_item = random.randint(1, self.item_num)
                negative_samples.append(neg_item)
                
            # 执行截断与左侧补零 (Padding)
            tokens_input = np.zeros(self.max_len, dtype=np.int64)
            seq_len = len(seq_input)
            if seq_len > 0:
                if seq_len >= self.max_len:
                    tokens_input[:] = seq_input[-self.max_len:]
                else:
                    tokens_input[-seq_len:] = seq_input
                    
            return torch.tensor(user, dtype=torch.long), \
                   torch.tensor(tokens_input, dtype=torch.long), \
                   torch.tensor(target_item, dtype=torch.long), \
                   torch.tensor(negative_samples, dtype=torch.long)