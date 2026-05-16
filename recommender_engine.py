"""
推荐引擎核心模块：从 inference.py 中抽象推荐逻辑，
提供 session 管理和 API 友好的推荐接口。
"""

import os
import uuid
import random
import threading

import torch
import pickle
import numpy as np
import torch.nn.functional as F

from sasrec_model import SASRec


class RecommenderEngine:
    """SASRec 推荐引擎，单例模式，模型只加载一次。"""

    def __init__(
        self,
        data_dir: str = "./sasrec_features",
        model_dir: str = "./saved_models",
        dict_path: str = "id2name.pkl",
    ):
        # 1. 检测计算设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 2. 加载 item2id.pkl (Raw ID -> Model ID)
        item2id_path = os.path.join(data_dir, "item2id.pkl")
        if not os.path.exists(item2id_path):
            raise FileNotFoundError(f"未找到字典文件: {item2id_path}")
        with open(item2id_path, "rb") as f:
            self.item2id = pickle.load(f)

        # 3. 加载 id2name.pkl (Raw ID -> 中文名)
        if os.path.exists(dict_path):
            with open(dict_path, "rb") as f:
                rawid2chinese = pickle.load(f)
        else:
            print("[Warning] 未找到 id2name.pkl，将默认使用原始 ID 展示。")
            rawid2chinese = {}

        # 4. 构建 model_id -> 中文名 的直接映射
        self.id2item = {}
        for raw_id, model_id in self.item2id.items():
            self.id2item[model_id] = rawid2chinese.get(str(raw_id), f"未知番剧_{raw_id}")

        self.item_num = len(self.item2id)

        # 5. 初始化 SASRec 模型
        self.max_seq_len = 50
        self.model = SASRec(
            item_num=self.item_num,
            max_seq_len=self.max_seq_len,
            hidden_units=50,
            num_heads=1,
            num_blocks=2,
            dropout_rate=0.0,
            device=self.device,
        ).to(self.device)

        # 6. 加载模型权重
        weight_path = os.path.join(model_dir, "sasrec_model_best.pth")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"未找到模型权重: {weight_path}")
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model.eval()

        # 7. 构建候选池
        self.valid_ids = [item_id for item_id in self.id2item.keys() if item_id != 0]
        self.popular_pool = sorted(self.valid_ids)[:500]

        # 8. Session 存储
        self._sessions: dict = {}
        self._lock = threading.Lock()

    # ========== 私有方法：抽取自 inference.py 的重复逻辑 ==========

    def _build_seq_tensor(self, user_sequence: list) -> torch.Tensor:
        """将用户序列构建为左填充的 (1, max_seq_len) 张量。"""
        seq_input = np.zeros(self.max_seq_len, dtype=np.int64)
        seq_len = len(user_sequence)
        if seq_len >= self.max_seq_len:
            seq_input[:] = user_sequence[-self.max_seq_len:]
        else:
            seq_input[-seq_len:] = user_sequence
        return torch.tensor(seq_input, dtype=torch.long).unsqueeze(0).to(self.device)

    def _score_items(self, user_sequence: list) -> torch.Tensor:
        """运行 SASRec 前向传播，返回 (item_num+1,) 的 logits 向量。"""
        seq_tensor = self._build_seq_tensor(user_sequence)
        with torch.no_grad():
            seq_out = self.model(seq_tensor)
            final_feat = seq_out[:, -1, :]
            logits = torch.matmul(
                final_feat, self.model.item_emb.weight.transpose(0, 1)
            ).squeeze(0)
        return logits

    def _mask_seen_items(self, logits: torch.Tensor, seen_items: set) -> torch.Tensor:
        """屏蔽 padding (id=0) 和所有已看过的 item。"""
        logits[0] = -1e9
        for seen_id in seen_items:
            logits[seen_id] = -1e9
        return logits

    def _format_item(self, item_id: int, tag: str = None) -> dict:
        """格式化单个 item 为 JSON 安全的字典。"""
        return {
            "item_id": int(item_id),
            "name": self.id2item.get(item_id, "未知番剧"),
            "tag": tag,
        }

    def _next_candidate(self, session_id: str) -> dict | None:
        """
        根据当前 session 状态，使用 E&E 策略选出下一个候选番剧。
        返回 _format_item 字典，候选耗尽时返回 None。
        """
        session = self._sessions[session_id]
        user_sequence = session["user_sequence"]
        seen_items = session["seen_items"]
        counter = session["smart_recommend_counter"]

        # 策略 1: 序列为空 → 热门初始
        if not user_sequence:
            item_id = self._pick_from_pool(seen_items)
            if item_id is None:
                return None
            tag = "热门初始"
            session["smart_recommend_counter"] = 0

        # 策略 2: 连续 3 次智能关联 → 随机探索
        elif counter >= 3:
            item_id = self._pick_from_pool(seen_items)
            if item_id is None:
                return None
            tag = "随机探索"
            session["smart_recommend_counter"] = 0

        # 策略 3: 智能关联
        else:
            logits = self._score_items(user_sequence)
            self._mask_seen_items(logits, seen_items)
            item_id = int(torch.argmax(logits).item())
            tag = "智能关联"
            session["smart_recommend_counter"] += 1

        return self._format_item(item_id, tag)

    def _pick_from_pool(self, seen_items: set) -> int | None:
        """从 popular_pool 中排除已看过的 item 后随机选一个，回退到 valid_ids。"""
        available = list(set(self.popular_pool) - seen_items)
        if not available:
            available = list(set(self.valid_ids) - seen_items)
        if not available:
            return None
        return random.choice(available)

    # ========== 公开方法 ==========

    def start_session(self) -> dict:
        """创建新推荐 session，返回初始状态和第一个候选。"""
        session_id = str(uuid.uuid4())
        session = {
            "session_id": session_id,
            "user_sequence": [],
            "seen_items": set(),
            "smart_recommend_counter": 0,
            "target_n": 5,
        }

        with self._lock:
            self._sessions[session_id] = session

        candidate = self._next_candidate(session_id)

        return {
            "session_id": session_id,
            "target_n": session["target_n"],
            "current_count": len(session["user_sequence"]),
            "candidate": candidate,
        }

    def submit_feedback(self, session_id: str, item_id: int, feedback: str) -> dict:
        """
        提交用户反馈，更新 session 状态，返回下一个候选。
        feedback: "like" 或 "dislike"
        """
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"会话不存在: {session_id}")
            session = self._sessions[session_id]

        if feedback == "like":
            session["user_sequence"].append(item_id)
            session["seen_items"].add(item_id)
        elif feedback == "dislike":
            session["seen_items"].add(item_id)
        else:
            raise ValueError(f"非法反馈类型: {feedback}，仅支持 'like' 或 'dislike'")

        candidate = self._next_candidate(session_id)

        # 构建 liked_items 列表（set 不可 JSON 序列化）
        liked_items = [
            self._format_item(mid) for mid in session["user_sequence"]
        ]

        return {
            "session_id": session_id,
            "target_n": session["target_n"],
            "current_count": len(session["user_sequence"]),
            "can_recommend": len(session["user_sequence"]) >= session["target_n"],
            "liked_items": liked_items,
            "candidate": candidate,
        }

    def next_candidate(self, session_id: str) -> dict:
        """获取下一个候选番剧（用于主动获取，不改变 session 状态以外的内容）。"""
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"会话不存在: {session_id}")

        candidate = self._next_candidate(session_id)
        return {"candidate": candidate}

    def recommend(self, session_id: str) -> dict:
        """根据当前 session 生成最终推荐 (Top-10 + I2I 关联)。"""
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"会话不存在: {session_id}")
            session = self._sessions[session_id]

        if not session["user_sequence"]:
            raise ValueError("未提供有效序列，无法生成推荐。")

        # 生成 Top-10
        logits = self._score_items(session["user_sequence"])
        self._mask_seen_items(logits, session["seen_items"])

        with torch.no_grad():
            _, top_indices = torch.topk(logits, k=10)
            top_indices = top_indices.cpu().tolist()

        top10 = []
        for rank, mid in enumerate(top_indices, 1):
            top10.append({
                "rank": rank,
                "item_id": int(mid),
                "name": self.id2item.get(mid, "未知番剧"),
            })

        # 生成 I2I 关联推荐（基于 Top-3）
        top3_ids = top_indices[:3]
        related = []

        with torch.no_grad():
            all_item_emb = self.model.item_emb.weight
            norm_item_emb = F.normalize(all_item_emb, p=2, dim=1)

            for rank, target_id in enumerate(top3_ids, 1):
                source_item = {
                    "rank": rank,
                    "item_id": int(target_id),
                    "name": self.id2item.get(target_id, "未知番剧"),
                }

                target_vec = norm_item_emb[target_id].unsqueeze(0)
                sim_scores = torch.matmul(
                    target_vec, norm_item_emb.transpose(0, 1)
                ).squeeze(0)

                # 屏蔽自身、padding、已看过
                sim_scores[target_id] = -1.0
                sim_scores[0] = -1.0
                for seen_id in session["seen_items"]:
                    sim_scores[seen_id] = -1.0

                _, sim_top = torch.topk(sim_scores, k=5)
                sim_top = sim_top.cpu().tolist()

                similar_items = []
                for sim_rank, sim_id in enumerate(sim_top, 1):
                    similar_items.append({
                        "rank": sim_rank,
                        "item_id": int(sim_id),
                        "name": self.id2item.get(sim_id, "未知番剧"),
                    })

                related.append({
                    "source": source_item,
                    "items": similar_items,
                })

        liked_items = [
            self._format_item(mid) for mid in session["user_sequence"]
        ]

        return {
            "top10": top10,
            "related": related,
            "liked_items": liked_items,
            "seen_count": len(session["seen_items"]),
            "liked_count": len(session["user_sequence"]),
        }

    def get_liked_items(self, session_id: str) -> list:
        """获取 session 中用户喜欢的番剧列表。"""
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"会话不存在: {session_id}")
            session = self._sessions[session_id]
        return [self._format_item(mid) for mid in session["user_sequence"]]
