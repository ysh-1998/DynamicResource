#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流量预测模型, 包含4种架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionLSTM(nn.Module):
    """
    带注意力机制的LSTM模型
    """
    
    def __init__(self, input_dim, hidden_dim=128, output_dim=1, num_layers=3, dropout=0.2):
        super(AttentionLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # bidirectional
            num_heads=8,
            dropout=dropout
        )
        
        # 输出层
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
        # 残差连接的投影
        self.residual_projection = nn.Linear(input_dim, hidden_dim * 2)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()
        
        # 输入投影
        x_proj = self.input_projection(x)
        
        # LSTM
        lstm_out, _ = self.lstm(x_proj)
        # lstm_out shape: (batch_size, seq_len, hidden_dim * 2)
        
        # 残差连接
        residual = self.residual_projection(x)
        lstm_out = lstm_out + residual
        
        # 层归一化
        lstm_out = self.layer_norm(lstm_out)
        
        # 注意力机制
        # 转置为 (seq_len, batch_size, hidden_dim * 2)
        lstm_out_t = lstm_out.transpose(0, 1)
        attn_out, _ = self.attention(lstm_out_t, lstm_out_t, lstm_out_t)
        attn_out = attn_out.transpose(0, 1)
        # attn_out shape: (batch_size, seq_len, hidden_dim * 2)
        
        # 使用最后一个时间步的输出
        final_out = attn_out[:, -1, :]
        
        # 全连接层
        out = self.fc1(final_out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class ImprovedTransformer(nn.Module):
    """
    改进的Transformer模型，包含位置编码和更好的架构
    """
    
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=4, dropout=0.2):
        super(ImprovedTransformer, self).__init__()
        
        self.d_model = d_model
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 全局池化层
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 输出层
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model // 2, 1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        
        # 输入投影
        x = self.input_projection(x)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer需要 (seq_len, batch_size, d_model)
        x = x.transpose(0, 1)
        
        # Transformer编码
        x = self.transformer_encoder(x)
        
        # 转回 (batch_size, seq_len, d_model)
        x = x.transpose(0, 1)
        
        # 全局池化
        x = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        x = self.global_pool(x).squeeze(-1)  # (batch_size, d_model)
        
        # 输出层
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class PositionalEncoding(nn.Module):
    """
    位置编码模块
    """
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class HybridModel(nn.Module):
    """
    混合模型：结合LSTM和Transformer的优势
    """
    
    def __init__(self, input_dim, hidden_dim=128, d_model=128, output_dim=1, dropout=0.2):
        super(HybridModel, self).__init__()
        
        # LSTM分支
        self.lstm_branch = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        # Transformer分支
        self.input_projection = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 2,
            dropout=dropout
        )
        self.transformer_branch = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 融合层
        fusion_dim = hidden_dim * 2 + d_model
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        # LSTM分支
        lstm_out, _ = self.lstm_branch(x)
        lstm_features = lstm_out[:, -1, :]  # 使用最后时间步
        
        # Transformer分支
        x_proj = self.input_projection(x)
        x_t = x_proj.transpose(0, 1)
        transformer_out = self.transformer_branch(x_t)
        transformer_features = transformer_out.mean(dim=0)  # 时间维度平均
        
        # 特征融合
        combined = torch.cat([lstm_features, transformer_features], dim=1)
        output = self.fusion(combined)
        
        return output


class EnsembleModel(nn.Module):
    """
    集成模型：组合多个模型的预测
    """
    
    def __init__(self, input_dim, hidden_dim=128, output_dim=1, dropout=0.2):
        super(EnsembleModel, self).__init__()
        
        # 创建多个不同配置的模型
        self.models = nn.ModuleList([
            AttentionLSTM(input_dim, hidden_dim, output_dim, num_layers=2, dropout=dropout),
            AttentionLSTM(input_dim, hidden_dim//2, output_dim, num_layers=3, dropout=dropout),
            ImprovedTransformer(input_dim, d_model=hidden_dim, nhead=4, num_layers=2, dropout=dropout),
        ])
        
        # 学习权重
        self.weights = nn.Parameter(torch.ones(len(self.models)) / len(self.models))
        
    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # 加权平均
        outputs = torch.stack(outputs, dim=1)  # (batch_size, num_models, output_dim)
        weights = F.softmax(self.weights, dim=0).view(1, -1, 1)
        ensemble_output = (outputs * weights).sum(dim=1)
        
        return ensemble_output