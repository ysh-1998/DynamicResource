#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大模型流量预测训练脚本
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import os
import logging
from datetime import datetime
from typing import Dict, Tuple
from model_traffic import AttentionLSTM, ImprovedTransformer, HybridModel, EnsembleModel
import matplotlib.pyplot as plt

class TrafficDataset(Dataset):
    """
    流量数据数据集
    """
    
    def __init__(self, features: np.ndarray, labels: np.ndarray, sequence_length: int = 60):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        return x, y

class DataProcessor:
    """
    改进的数据处理类
    """
    
    def __init__(self):
        # 使用RobustScaler对异常值更鲁棒
        self.scaler = RobustScaler()
        # 不对标签进行标准化，或使用对数变换
        self.use_log_transform = True
        
    def load_simulation_data(self, data_dir: str = 'simulation_data') -> Dict[str, pd.DataFrame]:
        """
        加载仿真数据
        """
        data_files = {}
        
        if not os.path.exists(data_dir):
            print(f"⚠️ 仿真数据目录 {data_dir} 不存在")
            return data_files
            
        for filename in os.listdir(data_dir):
            if filename.endswith('.csv') and filename != 'combined_simulation_data.csv':
                filepath = os.path.join(data_dir, filename)
                scenario_name = filename.replace('.csv', '')
                data_files[scenario_name] = pd.read_csv(filepath)
                
        return data_files
    
    def extract_features_from_df(self, df: pd.DataFrame, scenario_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        从DataFrame提取特征和标签
        """
        features = []
        labels = []
        
        # 提取基础特征
        for idx in range(len(df)):
            row = df.iloc[idx]
            
            # 基础特征
            qps = row['qps']
            tokens_per_second = row['effective_tokens_per_second']
            memory_demand = row.get('memory_demand_gb', row.get('memory_demand', 0))
            gpu_demand = row.get('required_gpus', row.get('gpu_demand', 0))
            gpu_utilization = row['gpu_utilization']
            memory_utilization = row['memory_utilization']
            
            # 计算平均输入输出token（如果不存在，使用tokens_per_second估算）
            avg_input_tokens = row.get('avg_input_tokens', tokens_per_second / qps * 0.3 if qps > 0 else 100)
            avg_output_tokens = row.get('avg_output_tokens', tokens_per_second / qps * 0.7 if qps > 0 else 300)
            
            # 时间特征（从timestamp解析）
            if 'timestamp' in row:
                timestamp = pd.to_datetime(row['timestamp'])
                hour = timestamp.hour
                day_of_week = timestamp.dayofweek
                is_peak_hour = 1 if (9 <= hour <= 17) else 0
            else:
                hour = row.get('hour', 12)
                day_of_week = row.get('day_of_week', 3)
                is_peak_hour = row.get('is_peak_hour', 0)
            
            # 时间编码（周期性）
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            dow_sin = np.sin(2 * np.pi * day_of_week / 7)
            dow_cos = np.cos(2 * np.pi * day_of_week / 7)
            
            time_features = [hour_sin, hour_cos, dow_sin, dow_cos, is_peak_hour]
            
            # 滑动窗口统计特征
            window_size = 10
            if idx >= window_size:
                recent_qps = df['qps'].iloc[idx-window_size:idx].values
                recent_avg = np.mean(recent_qps)
                recent_std = np.std(recent_qps)
                recent_trend = (qps - recent_qps[0]) / (recent_qps[0] + 1e-8)
            else:
                recent_avg = qps
                recent_std = 0
                recent_trend = 0
            
            # 场景特征（从行数据中获取）
            if 'scenario' in row:
                current_scenario = row['scenario']
                # 解析模型类型和负载模式
                if '_' in current_scenario:
                    model_name, load_pattern = current_scenario.rsplit('_', 1)
                else:
                    model_name, load_pattern = 'unknown', current_scenario
            else:
                model_name, load_pattern = 'unknown', scenario_name
            
            # 负载模式特征（one-hot编码）
            pattern_features = [
                1 if load_pattern == 'Daily' else 0,
                1 if load_pattern == 'Weekly' else 0,
                1 if load_pattern == 'Burst' else 0,
                1 if load_pattern == 'Steady' else 0
            ]
            
            # 模型类型特征（one-hot编码）
            model_features = [
                1 if model_name == 'GPT-4' else 0,
                1 if model_name == 'Claude' else 0,
                1 if model_name == 'LLaMA' else 0,
                1 if model_name == 'ChatGLM' else 0
            ]
            
            # 组合特征向量
            feature_vector = [
                qps, tokens_per_second, memory_demand, gpu_demand,
                gpu_utilization, memory_utilization, avg_input_tokens, avg_output_tokens
            ] + time_features + [recent_avg, recent_std, recent_trend] + pattern_features + model_features
            
            # 添加交互特征
            feature_vector.extend([
                qps * gpu_utilization,  # QPS和GPU利用率的交互
                memory_demand * memory_utilization,  # 内存需求和利用率的交互
                tokens_per_second / (avg_output_tokens + 1)  # 吞吐量效率
            ])
            
            features.append(feature_vector)
            labels.append(qps)
        
        return np.array(features), np.array(labels)
    
    def prepare_training_data(self, data_dir: str = 'simulation_data', sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        准备完整的训练数据
        """
        # 直接读取合并的数据文件
        combined_file = os.path.join(data_dir, 'combined_simulation_data.csv')
        
        is_main = (os.environ.get('LOCAL_RANK', '0') == '0')
        if not os.path.exists(combined_file):
            if is_main:
                print(f"❌ 找不到合并数据文件: {combined_file}")
            return None, None, None, None
        
        if is_main:
            print(f"📊 加载合并数据文件: {combined_file}")
        df = pd.read_csv(combined_file)
        if is_main:
            print(f"  数据量: {len(df)} 条记录")
        
        # 提取特征和标签
        features, labels = self.extract_features_from_df(df, "combined")
        
        all_features = [features]
        all_labels = [labels]
        
        # 合并所有数据
        X = np.vstack(all_features)
        y = np.concatenate(all_labels)
        
        # 特征标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 标签处理
        if self.use_log_transform:
            # 使用对数变换处理标签，保持相对关系
            y_transformed = np.log1p(y)  # log(1+y) 避免log(0)
        else:
            y_transformed = y
        
        # 创建序列数据
        X_seq, y_seq = [], []
        for i in range(sequence_length, len(X_scaled)):
            X_seq.append(X_scaled[i-sequence_length:i])
            y_seq.append(y_transformed[i])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        # 时序分割（而不是随机分割）
        split_idx = int(len(X_seq) * 0.8)
        X_train = X_seq[:split_idx]
        X_val = X_seq[split_idx:]
        y_train = y_seq[:split_idx]
        y_val = y_seq[split_idx:]
        
        if is_main:
            print(f"📊 训练集形状: {X_train.shape}, 验证集形状: {X_val.shape}")
            print(f"📊 标签统计 - 训练集: min={np.expm1(y_train.min()):.2f}, max={np.expm1(y_train.max()):.2f}, mean={np.expm1(y_train.mean()):.2f}")
            print(f"📊 标签统计 - 验证集: min={np.expm1(y_val.min()):.2f}, max={np.expm1(y_val.max()):.2f}, mean={np.expm1(y_val.mean()):.2f}")
        
        return X_train, X_val, y_train, y_val
    
    def inverse_transform_labels(self, y_transformed):
        """
        反变换标签
        """
        if self.use_log_transform:
            return np.expm1(y_transformed)  # exp(y) - 1
        else:
            return y_transformed

class ModelTrainer:
    """
    改进的模型训练器
    """
    
    def __init__(self, model_type: str = 'lstm', device: str = 'cpu', gpu_ids: str = None):
        self.model_type = model_type
        self.device = device
        self.gpu_ids = gpu_ids
        self.accelerator = None
        self.data_processor = DataProcessor()
        
    def train(self, 
              data_dir: str = 'simulation_data',
              sequence_length: int = 60,
              batch_size: int = 32,
              epochs: int = 100,
              learning_rate: float = 0.001,
              weight_decay: float = 0.0001,
              early_stopping_patience: int = 15) -> Dict:
        """
        训练主函数
        """
        # Accelerator 初始化后才能用 is_main_process，这里先用条件判断
        print("🚀 开始训练流程...")
        
        # 准备数据
        X_train, X_val, y_train, y_val = self.data_processor.prepare_training_data(data_dir, sequence_length)
        
        if X_train is None:
            return None
        
        # 创建数据集
        train_dataset = TrafficDataset(X_train, y_train, sequence_length)
        val_dataset = TrafficDataset(X_val, y_val, sequence_length)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 获取特征维度
        feature_dim = X_train.shape[2]
        
        # 创建模型
        if self.model_type == 'lstm':
            model = AttentionLSTM(
                input_dim=feature_dim,
                hidden_dim=128,
                output_dim=1,
                num_layers=3,
                dropout=0.2
            )
        elif self.model_type == 'transformer':
            model = ImprovedTransformer(
                input_dim=feature_dim,
                d_model=128,
                nhead=8,
                num_layers=4,
                dropout=0.2
            )
        elif self.model_type == 'hybrid':
            model = HybridModel(
                input_dim=feature_dim,
                hidden_dim=128,
                d_model=128,
                output_dim=1,
                dropout=0.2
            )
        elif self.model_type == 'ensemble':
            model = EnsembleModel(
                input_dim=feature_dim,
                hidden_dim=128,
                output_dim=1,
                dropout=0.2
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # 优化器
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # 损失函数 - 使用Huber Loss对异常值更鲁棒
        criterion = nn.HuberLoss(delta=1.0)
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
        
        # 初始化accelerator
        if self.gpu_ids:
            gpu_list = [int(x.strip()) for x in self.gpu_ids.split(',')]
            os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_ids
        self.accelerator = Accelerator()
        if self.gpu_ids and self.accelerator.is_main_process:
            print(f"🔧 使用GPU: {gpu_list}")
        
        # 准备accelerator
        model, optimizer, train_loader, val_loader = self.accelerator.prepare(
            model, optimizer, train_loader, val_loader
        )
        
        # 初始化logging（仅主进程写文件）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs('logs', exist_ok=True)
        log_path = f'logs/training_{self.model_type}_{timestamp}.log'
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            fmt = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            if self.accelerator.is_main_process:
                logger.addHandler(logging.FileHandler(log_path))
            logger.addHandler(logging.StreamHandler())
            for h in logger.handlers:
                h.setFormatter(fmt)
        
        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        os.makedirs('models', exist_ok=True)
        model_path = f'models/{self.model_type}_traffic_{timestamp}.pth'
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                
                self.accelerator.backward(loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            # 验证阶段
            model.eval()
            val_loss = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = model(batch_x).squeeze()
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    all_preds.extend(outputs.cpu().numpy())
                    all_labels.extend(batch_y.cpu().numpy())
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # ── 跨进程聚合预测结果，确保所有进程指标一致 ──────────────────────
            # gather_for_metrics 会自动对齐各进程的张量并去除 padding
            all_preds_tensor = torch.tensor(all_preds, device=self.accelerator.device)
            all_labels_tensor = torch.tensor(all_labels, device=self.accelerator.device)
            all_preds_gathered = self.accelerator.gather_for_metrics(all_preds_tensor).cpu().numpy()
            all_labels_gathered = self.accelerator.gather_for_metrics(all_labels_tensor).cpu().numpy()
            
            # 跨进程同步 val_loss（取各进程均值），保证学习率调度和早停判断一致
            val_loss_tensor = torch.tensor(avg_val_loss, device=self.accelerator.device)
            avg_val_loss = self.accelerator.reduce(val_loss_tensor, reduction="mean").item()

            # 计算真实值的评估指标（基于全量验证集）
            preds_real = self.data_processor.inverse_transform_labels(all_preds_gathered)
            labels_real = self.data_processor.inverse_transform_labels(all_labels_gathered)

            # 确保预测值非负
            preds_real = np.maximum(preds_real, 0)
            
            # 计算MAPE
            mask = labels_real > 1.0  # 只在QPS > 1的情况下计算MAPE
            if mask.any():
                mape = np.mean(np.abs((labels_real[mask] - preds_real[mask]) / labels_real[mask])) * 100
            else:
                mape = 100.0
            
            # 计算RMSE
            rmse = np.sqrt(np.mean((labels_real - preds_real) ** 2))
            
            # 记录
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            # 学习率调度（所有进程使用相同的 avg_val_loss，保持调度器同步）
            scheduler.step(avg_val_loss)
            
            # ── 所有进程在此同步，再做早停/保存判断，避免 NCCL 超时 ──────────
            self.accelerator.wait_for_everyone()

            # 早停
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # 保存最佳模型（仅主进程写文件，其余进程在后面的 wait 处等待）
                if self.accelerator.is_main_process:
                    torch.save({
                        'model_state_dict': self.accelerator.unwrap_model(model).state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'feature_scaler': self.data_processor.scaler,
                        'use_log_transform': self.data_processor.use_log_transform,
                        'feature_dim': feature_dim,
                        'best_val_loss': best_val_loss,
                        'config': {
                            'model_type': self.model_type,
                            'sequence_length': sequence_length
                        }
                    }, model_path)
                # 等待主进程写完文件后，所有进程再继续
                self.accelerator.wait_for_everyone()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if self.accelerator.is_main_process:
                        logger.info(f"⚠️ 早停触发！验证损失在{early_stopping_patience}个epoch内没有改善")
                    break
            
            # 记录指标（仅主进程）
            current_lr = optimizer.param_groups[0]['lr']
            if self.accelerator.is_main_process:
                logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, MAPE: {mape:.2f}%, RMSE: {rmse:.2f}, LR: {current_lr}")
                logger.info(f"  QPS范围 - 真实: [{labels_real.min():.2f}, {labels_real.max():.2f}], 预测: [{preds_real.min():.2f}, {preds_real.max():.2f}]")
        
        # 结束训练 —— 所有进程在此同步，避免主进程做耗时 I/O 时其他进程
        # 还在等待 NCCL 集合通信（如 BROADCAST）而超时崩溃
        self.accelerator.wait_for_everyone()
        
        if self.accelerator.is_main_process:
            logger.info(f"✅ 训练完成！最佳验证损失: {best_val_loss:.6f}")
        
        # 绘制训练曲线
        if self.accelerator.is_main_process:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training History')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.scatter(labels_real[:1000], preds_real[:1000], alpha=0.5)
            plt.plot([labels_real.min(), labels_real.max()], 
                     [labels_real.min(), labels_real.max()], 'r--', lw=2)
            plt.xlabel('True QPS')
            plt.ylabel('Predicted QPS')
            plt.title('Prediction vs Truth (First 1000 samples)')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f'training_results_{self.model_type}_{timestamp}.png', dpi=150)
            plt.close()
        
        return {
            'best_val_loss': best_val_loss,
            'final_mape': mape,
            'final_rmse': rmse,
            'model_path': model_path if 'model_path' in locals() else None
        }

def main():
    """
    主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='训练流量预测模型')
    parser.add_argument('--model', type=str, default='lstm',
                        choices=['lstm', 'transformer', 'hybrid', 'ensemble'],
                        help='模型类型')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--sequence-length', type=int, default=360, help='序列长度')
    parser.add_argument('--gpu-ids', type=str, default=None, help='GPU IDs (e.g., "0,1,2")')
    
    args = parser.parse_args()
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() and args.gpu_ids else 'cpu'
    
    # 创建训练器
    trainer = ModelTrainer(
        model_type=args.model,
        device=device,
        gpu_ids=args.gpu_ids
    )
    
    # 开始训练
    results = trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        sequence_length=args.sequence_length
    )
    
    if results:
        print(f"🎉 训练完成！")
        print(f"模型已保存到: {results['model_path']}")
        print(f"最终MAPE: {results['final_mape']:.2f}%")
        print(f"最终RMSE: {results['final_rmse']:.2f}")

if __name__ == '__main__':
    main()