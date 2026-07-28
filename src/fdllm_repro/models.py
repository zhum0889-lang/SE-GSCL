"""Stage-1 FD-LLM reproduction models.

This module implements the data encoder, frozen text prototypes, contrastive
alignment loss, and fuzzy semantic embedding from the paper.

中文说明：
本文件主要实现 FD-LLM 复现中的第一阶段：
1. 将一维轴承振动窗口编码为 data embedding；
2. 将 data embedding 对齐到故障文本原型空间；
3. 根据 data/text 相似度计算 Fuzzy Semantic Embedding (FSE)。

这一阶段是后续接入 LLM 的基础。只有 data embedding 已经进入文本语义空间，
后面的 Data/FSE embedding 注入 LLM 才有意义。
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.feature_extraction.text import HashingVectorizer
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


class LSTMSignalEncoder(nn.Module):
    """基础 LSTM 信号编码器。

    这是最早的轻量版本：直接把一维振动序列输入 BiLSTM，
    再映射到文本 embedding 维度。当前主实验更多使用 ConvLSTM。
    """

    def __init__(self, embed_dim: int = 128, hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        _, (h_n, _) = self.lstm(x)
        if self.lstm.bidirectional:
            state = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            state = h_n[-1]
        emb = self.proj(state)
        return F.normalize(emb, dim=-1)


class ConvLSTMSignalEncoder(nn.Module):
    """ConvLSTM 信号编码器。

    输入：
        x: [batch, window_size] 或 [batch, window_size, 1]

    输出：
        emb: [batch, embed_dim]

    设计思路：
    - 前面的 Conv1d stem 负责提取局部冲击/波形模式，并降低序列长度；
    - BiLSTM 负责建模降采样后的时序依赖；
    - Linear projection 将信号特征映射到 LLM 文本 embedding 维度；
    - 最后做 L2 normalize，便于和文本原型做余弦相似度。
    """

    def __init__(
        self,
        embed_dim: int = 128,
        hidden_dim: int = 64,
        conv_dim: int = 32,
        input_channels: int = 1,
    ):
        super().__init__()
        # Conv stem 将原始 1024 点振动窗口压缩成更短的特征序列，
        # 这样 LSTM 训练更快，也更容易捕捉局部故障冲击。
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, conv_dim // 2, kernel_size=9, stride=4, padding=4),
            nn.BatchNorm1d(conv_dim // 2),
            nn.GELU(),
            nn.Conv1d(conv_dim // 2, conv_dim, kernel_size=9, stride=4, padding=4),
            nn.BatchNorm1d(conv_dim),
            nn.GELU(),
        )
        # 双向 LSTM 同时利用前后时序上下文。
        self.lstm = nn.LSTM(
            input_size=conv_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        # 投影到文本原型空间。embed_dim 在 Qwen 接入实验中为 896。
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 统一输入形状为 Conv1d 需要的 [batch, channel, length]。
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim == 3 and x.shape[-1] == 1:
            x = x.transpose(1, 2)

        # Conv 输出 [batch, channel, reduced_length]，
        # LSTM 需要 [batch, reduced_length, channel]。
        feat = self.stem(x).transpose(1, 2)
        _, (h_n, _) = self.lstm(feat)
        state = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        emb = self.proj(state)

        # 归一化后，data_emb @ text_emb.T 等价于余弦相似度。
        return F.normalize(emb, dim=-1)


def encode_text_descriptions(
    descriptions: list[str],
    embed_dim: int,
    class_anchor_weight: float = 0.65,
) -> np.ndarray:
    """Offline frozen text encoder.

    A local LLM embedding layer can replace this later. For the first runnable
    reproduction, hashed text features plus deterministic class anchors give us
    stable semantic prototypes without network downloads. The anchors avoid the
    collapse that happens when many fault descriptions differ by only a few
    tokens such as 0.007/0.014/0.021.
    """

    vectorizer = HashingVectorizer(
        n_features=2048,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
        ngram_range=(1, 2),
    )
    hashed = vectorizer.transform(descriptions).astype(np.float32).toarray()
    seed = int(hashlib.sha256("fdllm-text-projection".encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    proj = rng.normal(0.0, 1.0 / math.sqrt(hashed.shape[1]), size=(hashed.shape[1], embed_dim))
    semantic_embs = hashed @ proj
    anchor_embs = _orthogonal_anchors(len(descriptions), embed_dim)
    embs = (1.0 - class_anchor_weight) * semantic_embs + class_anchor_weight * anchor_embs
    norm = np.linalg.norm(embs, axis=1, keepdims=True)
    return (embs / np.maximum(norm, 1e-8)).astype(np.float32)


def _orthogonal_anchors(n_items: int, embed_dim: int) -> np.ndarray:
    if embed_dim >= n_items:
        anchors = np.zeros((n_items, embed_dim), dtype=np.float32)
        anchors[:, :n_items] = np.eye(n_items, dtype=np.float32)
        return anchors

    seed = int(hashlib.sha256("fdllm-class-anchors".encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    anchors = rng.normal(size=(n_items, embed_dim)).astype(np.float32)
    anchors /= np.maximum(np.linalg.norm(anchors, axis=1, keepdims=True), 1e-8)
    return anchors


def prototype_alignment_loss(
    data_emb: torch.Tensor,
    text_bank: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """InfoNCE alignment against one frozen text prototype per class.

    The paper describes CLIP-style data-text pairs. In this runnable milestone
    we use one text description per class, so a full pairwise batch loss would
    create false negatives whenever two samples share a class. This class
    prototype form preserves the same alignment objective without that issue.
    """

    # data_emb 和 text_bank 都已 L2 normalize，
    # 点积就是余弦相似度；temperature 控制分类分布的尖锐程度。
    logits = data_emb @ text_bank.T / temperature
    return F.cross_entropy(logits, labels)


@dataclass
class TrainHistory:
    epoch: int
    loss: float
    alignment_acc: float


def train_alignment(
    encoder: LSTMSignalEncoder,
    x_train: np.ndarray,
    y_train: np.ndarray,
    text_embeddings: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    temperature: float = 0.07,
) -> list[TrainHistory]:
    """训练数据编码器，使振动窗口靠近对应类别的文本原型。

    注意：此阶段只训练 encoder，不训练文本原型和 LLM。
    text_embeddings 相当于冻结的语义类别中心。
    """

    encoder.to(device)
    text_bank = torch.tensor(text_embeddings, dtype=torch.float32, device=device)
    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.long)
    loader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(encoder.parameters(), lr=learning_rate, weight_decay=1e-4)

    history: list[TrainHistory] = []
    for epoch in range(1, epochs + 1):
        encoder.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            data_emb = encoder(xb)

            # 类原型对齐：每个样本与 19 个文本原型计算相似度，
            # 用真实类别做交叉熵监督。
            loss = prototype_alignment_loss(data_emb, text_bank, yb, temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                # alignment_acc 不是最终测试准确率，只是训练过程中
                # data embedding 最近文本原型是否为真实类别的监控指标。
                logits = data_emb @ text_bank.T
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += int(yb.numel())
                total_loss += float(loss.item()) * int(yb.numel())

        history.append(TrainHistory(epoch, total_loss / total, correct / total))
    return history


@torch.no_grad()
def fuzzy_semantic_predict(
    encoder: LSTMSignalEncoder,
    x: np.ndarray,
    text_embeddings: np.ndarray,
    device: str,
    temperature: float = 0.5,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算 Fuzzy Semantic Embedding。

    返回：
    - probs: 每个样本在所有文本原型上的 softmax 概率；
    - fuzzy_emb: 用 probs 对文本原型加权得到的语义向量；
    - data_emb: 数据编码器输出的原始信号向量。

    FSE 的关键思想是“不只取 top-1 类别”，而是保留类别间模糊关系。
    例如某个样本可能同时接近 Ball_014 和 Ball_021，FSE 会把这种不确定性编码进去。
    """

    encoder.eval()
    encoder.to(device)
    text_bank = torch.tensor(text_embeddings, dtype=torch.float32, device=device)
    loader = DataLoader(torch.tensor(x, dtype=torch.float32), batch_size=batch_size, shuffle=False)

    all_probs: list[np.ndarray] = []
    all_semantic: list[np.ndarray] = []
    all_data: list[np.ndarray] = []
    for xb in loader:
        xb = xb.to(device)
        data_emb = encoder(xb)
        logits = data_emb @ text_bank.T / temperature
        probs = F.softmax(logits, dim=1)

        # FSE = sum_k P(class=k | signal) * text_embedding_k
        # 因此 fuzzy_emb 仍然处在 LLM 文本语义空间中，可以继续注入 LLM。
        fuzzy_emb = probs @ text_bank
        all_probs.append(probs.cpu().numpy())
        all_semantic.append(fuzzy_emb.cpu().numpy())
        all_data.append(data_emb.cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_semantic), np.concatenate(all_data)
