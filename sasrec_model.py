import torch
import torch.nn as nn

class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        # inputs shape: (batch_size, seq_len, hidden_units)
        outputs = self.dropout1(self.relu(self.conv1(inputs.transpose(-1, -2))))
        outputs = self.dropout2(self.conv2(outputs))
        # outputs shape: (batch_size, seq_len, hidden_units)
        return outputs.transpose(-1, -2) + inputs

class SASRecBlock(nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super(SASRecBlock, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_units, 
            num_heads=num_heads, 
            dropout=dropout_rate, 
            batch_first=True
        )
        self.layer_norm1 = nn.LayerNorm(hidden_units, eps=1e-8)
        self.ffn = PointWiseFeedForward(hidden_units, dropout_rate)
        self.layer_norm2 = nn.LayerNorm(hidden_units, eps=1e-8)

    def forward(self, seqs, attention_mask):
        # Q, K, V 均为 seqs
        # attention_mask 用于屏蔽未来的信息以及 Padding 的 0
        attn_outputs, _ = self.attention(
            query=seqs, 
            key=seqs, 
            value=seqs, 
            attn_mask=attention_mask,
            need_weights=False
        )
        
        # 残差连接与 LayerNorm
        seqs = self.layer_norm1(seqs + attn_outputs)
        
        # 前馈神经网络与第二个 LayerNorm
        ffn_outputs = self.ffn(seqs)
        seqs = self.layer_norm2(seqs + ffn_outputs)
        
        return seqs

class SASRec(nn.Module):
    def __init__(self, item_num, max_seq_len, hidden_units, num_heads, num_blocks, dropout_rate, device):
        super(SASRec, self).__init__()
        self.item_num = item_num
        self.max_seq_len = max_seq_len
        self.hidden_units = hidden_units
        self.device = device
        
        # 物品 Embedding (预留 0 作为 Padding，所以容量是 item_num + 1)
        self.item_emb = nn.Embedding(item_num + 1, hidden_units, padding_idx=0)
        # 可学习的位置 Embedding
        self.pos_emb = nn.Embedding(max_seq_len, hidden_units)
        self.emb_dropout = nn.Dropout(p=dropout_rate)

        # 堆叠多个 Self-Attention Block
        self.attention_blocks = nn.ModuleList([
            SASRecBlock(hidden_units, num_heads, dropout_rate) for _ in range(num_blocks)
        ])

    def forward(self, log_seqs):
        # log_seqs shape: (batch_size, max_seq_len)
        
        # 生成基于序列实际物品的 Mask (屏蔽所有的 Padding 0)
        # shape: (batch_size, max_seq_len)
        timeline_mask = (log_seqs == 0)
        
        # 获取当前 Batch 的序列长度
        seq_len = log_seqs.shape[1]
        
        # 提取物品 Embedding
        seqs = self.item_emb(log_seqs)
        
        # 生成位置索引并提取位置 Embedding
        positions = torch.arange(seq_len, dtype=torch.long, device=self.device)
        positions = positions.unsqueeze(0).expand_as(log_seqs)
        pos_embeddings = self.pos_emb(positions)
        
        # 物品与位置 Embedding 相加
        seqs = seqs + pos_embeddings
        
        # Padding Mask: 将 Padding 所在位置的 Embedding 置为 0
        seqs = seqs * (~timeline_mask.unsqueeze(-1)).float()
        seqs = self.emb_dropout(seqs)

        # 构建因果掩码 (Causality Mask)
        # 上三角矩阵设为 True，表示 t 时刻不能看到 t 时刻之后的内容
        causality_mask = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=self.device), 
            diagonal=1
        )

        # 依次通过堆叠的 Attention Blocks
        for block in self.attention_blocks:
            seqs = block(seqs, causality_mask)
            # 在每一层之后再次确保 Padding 位置的输出严格为 0
            seqs = seqs * (~timeline_mask.unsqueeze(-1)).float()

        # seqs 此时就是用户在每个时间步 t 融合了历史信息的综合兴趣表征
        return seqs