#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大模型算力资源调度仿真数据生成器 - 改进版

基于真实的LLM部署经验重新设计，生成更符合实际的仿真数据
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Tuple
import random

# 设置matplotlib参数
plt.rcParams['axes.unicode_minus'] = False

class ResourceSimulator:
    """
    算力资源调度仿真数据生成器
    基于真实场景的参数配置
    """
    
    def __init__(self, seed: int = 42):
        """
        初始化仿真器
        
        Args:
            seed: 随机种子，确保结果可重现
        """
        np.random.seed(seed)
        random.seed(seed)
        
        # 大模型配置参数（基于真实部署经验）
        self.model_configs = {
            'GPT-4': {
                'model_size_b': 1800,  # 模型参数量（十亿）- 16 expert MOE，总参数1.8T
                'tokens_per_second_per_gpu': {'A100': 800, 'H100': 1500},  # 不同GPU的吞吐量
                'memory_per_billion_params': 2.5,   # 每十亿参数需要的显存(GB)，FP16
                'batch_memory_factor': 1.5,         # 批处理时的内存放大系数
                'gpu_efficiency': 0.65,             # 实际GPU利用率（大模型效率更低）
                'min_gpus_per_replica': 32,          # 最小GPU数（模型并行）
                'context_length': 128000,             # 上下文长度
                'prefill_compute_factor': 2.2       # prefill阶段的计算倍数
            },
            'ChatGLM': {
                'model_size_b': 130,
                'tokens_per_second_per_gpu': {'A100': 2500, 'H100': 4000},  # 不同GPU的吞吐量
                'memory_per_billion_params': 2.5,
                'batch_memory_factor': 1.4,
                'gpu_efficiency': 0.80,
                'min_gpus_per_replica': 4,
                'context_length': 32768,  # ChatGLM-130B支持32K上下文
                'prefill_compute_factor': 1.8
            },
            'Claude': {
                'model_size_b': 400,  # Claude 3.5 Sonnet 约400B参数
                'tokens_per_second_per_gpu': {'A100': 1800, 'H100': 3000},  # 不同GPU的吞吐量
                'memory_per_billion_params': 2.5,
                'batch_memory_factor': 1.45,
                'gpu_efficiency': 0.75,  # 长上下文效率较低
                'min_gpus_per_replica': 8,
                'context_length': 200000,  # Claude的超长上下文
                'prefill_compute_factor': 3.0  # 长上下文prefill计算更密集
            },
            'LLaMA': {
                'model_size_b': 70,
                'tokens_per_second_per_gpu': {'A100': 3000, 'H100': 5000},  # 不同GPU的吞吐量
                'memory_per_billion_params': 2.5,
                'batch_memory_factor': 1.3,
                'gpu_efficiency': 0.85,  # 较小模型效率更高
                'min_gpus_per_replica': 2,
                'context_length': 4096,
                'prefill_compute_factor': 1.5
            }
        }
        
        # 硬件配置（真实的GPU服务器配置）
        self.hardware_configs = {
            'server_specs': {
                'cpu_cores': 64,
                'memory_gb': 512,
                'gpu_slots': 8,
                'network_bandwidth_gbps': 100
            },
            'gpu_specs': {
                'A100': {
                    'memory_gb': 80, 
                    'tflops': 312,  # FP16 tensor core性能
                    'memory_bandwidth_gb': 2039,
                    'cost_per_hour': 2.5,
                    'power_watts': 400
                },
                'H100': {
                    'memory_gb': 80, 
                    'tflops': 1000,
                    'memory_bandwidth_gb': 3352,
                    'cost_per_hour': 4.0,
                    'power_watts': 700
                }
            },
            # SLA配置
            'sla_configs': {
                'p50_latency_ms': 100,
                'p95_latency_ms': 500,
                'p99_latency_ms': 1000,
                'availability_target': 0.999
            }
        }
    
    def generate_traffic_pattern(self, 
                               duration_hours: int = 24*7, 
                               base_qps: float = 100.0,
                               pattern_type: str = 'daily',
                               model_type: str = None) -> pd.DataFrame:
        """
        生成流量模式数据，包含更真实的特征
        """
        timestamps = pd.date_range(
            start=datetime.now(),
            periods=duration_hours * 60,  # 每分钟一个数据点
            freq='1min'
        )
        
        qps_values = []
        avg_input_tokens = []
        avg_output_tokens = []
        
        for _, ts in enumerate(timestamps):
            hour = ts.hour
            day_of_week = ts.weekday()
            minute_of_day = hour * 60 + ts.minute
            
            # 基础QPS计算
            if pattern_type == 'daily':
                # 日常模式：工作时间高峰
                if 9 <= hour <= 18:  # 工作时间
                    peak_factor = 1.5
                elif 19 <= hour <= 22:  # 晚高峰
                    peak_factor = 1.2
                elif 7 <= hour <= 8 or hour == 23:  # 早晚过渡
                    peak_factor = 0.8
                else:  # 深夜
                    peak_factor = 0.3
                qps = base_qps * peak_factor
                
            elif pattern_type == 'weekly':
                # 周模式：工作日vs周末
                weekly_factor = 0.6 if day_of_week >= 5 else 1.0
                daily_factor = 0.3 + 0.7 * (1 + np.sin((hour - 6) * np.pi / 12)) / 2
                qps = base_qps * weekly_factor * daily_factor
                
            elif pattern_type == 'burst':
                # 突发模式：模拟热点事件
                base_factor = 0.5 + 0.3 * np.sin(minute_of_day * 2 * np.pi / (24 * 60))
                # 热点事件概率和强度
                if np.random.random() < 0.02:  # 2%概率发生热点
                    # burst_duration = np.random.randint(5, 30)  # 持续5-30分钟
                    burst_intensity = np.random.uniform(2, 8)  # 2-8倍基础流量
                    qps = base_qps * base_factor * burst_intensity
                else:
                    qps = base_qps * base_factor
                    
            else:  # steady
                # 稳定模式：小幅波动 ±8%
                noise = np.random.uniform(-0.08, 0.08)
                qps = base_qps * (1 + noise)
            
            # 添加真实的随机波动
            qps *= (1 + np.random.normal(0, 0.1))
            qps = max(1, qps)  # 确保至少1 QPS
            
            # 输入输出token分布（基于真实使用场景和模型类型）
            if model_type == 'GPT-4':
                # GPT-4: 企业级复杂任务
                if 9 <= hour <= 18:  # 工作时间
                    avg_input = np.random.normal(2000, 500)
                    avg_output = np.random.normal(3000, 800)
                else:
                    avg_input = np.random.normal(1000, 300)
                    avg_output = np.random.normal(1500, 500)
            elif model_type == 'Claude':
                # Claude: 长上下文研究任务
                if pattern_type == 'weekly' and day_of_week < 5:  # 工作日
                    avg_input = np.random.normal(50000, 20000)  # 研究论文等长文档
                    avg_output = np.random.normal(5000, 2000)   # 总结输出
                else:
                    avg_input = np.random.normal(20000, 10000)
                    avg_output = np.random.normal(3000, 1000)
            elif model_type == 'ChatGLM':
                # ChatGLM: 消费级对话
                if pattern_type == 'burst' and np.random.random() < 0.3:  # 热点时段
                    avg_input = np.random.normal(300, 100)
                    avg_output = np.random.normal(500, 200)
                else:
                    avg_input = np.random.normal(200, 80)
                    avg_output = np.random.normal(400, 150)
            elif model_type == 'LLaMA':
                # LLaMA: API服务，相对稳定
                avg_input = np.random.normal(400, 150)
                avg_output = np.random.normal(600, 200)
            else:
                # 默认值
                if pattern_type == 'daily' and 9 <= hour <= 18:
                    avg_input = np.random.normal(500, 200)
                    avg_output = np.random.normal(800, 300)
                else:
                    avg_input = np.random.normal(200, 100)
                    avg_output = np.random.normal(400, 200)
            
            # 确保正值
            avg_input = max(50, avg_input)
            avg_output = max(50, avg_output)
            
            qps_values.append(qps)
            avg_input_tokens.append(avg_input)
            avg_output_tokens.append(avg_output)
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'qps': qps_values,
            'avg_input_tokens': avg_input_tokens,
            'avg_output_tokens': avg_output_tokens
        })
    
    def calculate_resource_requirements(self, 
                                      traffic_df: pd.DataFrame,
                                      model_type: str = 'GPT-4',
                                      gpu_type: str = 'A100',
                                      redundancy_factor: float = 1.3) -> pd.DataFrame:
        """
        根据流量计算资源需求（基于真实部署经验）
        """
        model_config = self.model_configs[model_type]
        gpu_config = self.hardware_configs['gpu_specs'][gpu_type]
        server_config = self.hardware_configs['server_specs']
        
        results = []
        
        for _, row in traffic_df.iterrows():
            qps = row['qps']
            timestamp = row['timestamp']
            avg_input_tokens = row['avg_input_tokens']
            avg_output_tokens = row['avg_output_tokens']
            
            # 1. 计算吞吐量需求
            # Prefill阶段（处理输入）和Generation阶段（生成输出）
            prefill_tokens = qps * avg_input_tokens
            generation_tokens = qps * avg_output_tokens
            
            # 考虑prefill计算密集度更高
            effective_tokens = (prefill_tokens * model_config['prefill_compute_factor'] + 
                              generation_tokens)
            
            # 2. 计算内存需求
            # 模型权重内存
            model_memory = model_config['model_size_b'] * model_config['memory_per_billion_params']
            
            # KV cache内存（每个请求需要存储注意力缓存）
            # 估算：batch_size * seq_len * hidden_dim * num_layers * 2 (K和V) * bytes_per_param
            concurrent_requests = qps * 2  # 假设平均处理时间2秒
            kv_cache_per_request = (avg_input_tokens + avg_output_tokens) * 0.001  # GB，简化计算
            kv_cache_total = concurrent_requests * kv_cache_per_request
            
            # 激活值内存和其他开销
            activation_memory = model_memory * 0.2  # 约20%的模型大小
            
            total_memory_per_replica = (model_memory + kv_cache_total + activation_memory) * model_config['batch_memory_factor']
            
            # 3. 计算GPU需求
            # 基于吞吐量
            if isinstance(model_config['tokens_per_second_per_gpu'], dict):
                tokens_per_gpu_base = model_config['tokens_per_second_per_gpu'][gpu_type]
            else:
                tokens_per_gpu_base = model_config['tokens_per_second_per_gpu']
            tokens_per_gpu = tokens_per_gpu_base * model_config['gpu_efficiency']
            # required_gpus_throughput = np.ceil(effective_tokens / tokens_per_gpu)
            
            # 基于内存（考虑模型并行）
            gpus_per_replica = max(
                model_config['min_gpus_per_replica'],
                np.ceil(total_memory_per_replica / gpu_config['memory_gb'])
            )
            
            # 计算需要多少个副本来处理吞吐量
            tokens_per_replica = gpus_per_replica * tokens_per_gpu
            required_replicas = np.ceil(effective_tokens / tokens_per_replica)
            
            # 总GPU数（包含冗余）
            required_gpus = int(required_replicas * gpus_per_replica * redundancy_factor)
            
            # 4. 计算服务器需求
            gpus_per_server = server_config['gpu_slots']
            required_servers = int(np.ceil(required_gpus / gpus_per_server))
            
            # 5. 成本计算
            hourly_cost = required_gpus * gpu_config['cost_per_hour']
            
            # 6. 资源利用率
            actual_throughput = effective_tokens
            max_throughput = required_gpus * tokens_per_gpu
            gpu_utilization = min(actual_throughput / max_throughput, 1.0) if max_throughput > 0 else 0
            
            actual_memory = required_replicas * total_memory_per_replica
            max_memory = required_gpus * gpu_config['memory_gb']
            memory_utilization = min(actual_memory / max_memory, 1.0) if max_memory > 0 else 0
            
            # 7. 延迟估算（简化）
            queue_depth = concurrent_requests / required_replicas
            estimated_latency_ms = 50 + (avg_input_tokens + avg_output_tokens) * 0.5 + queue_depth * 10
            
            results.append({
                'timestamp': timestamp,
                'qps': qps,
                'avg_input_tokens': avg_input_tokens,
                'avg_output_tokens': avg_output_tokens,
                'effective_tokens_per_second': effective_tokens,
                'concurrent_requests': concurrent_requests,
                'model_memory_gb': model_memory,
                'kv_cache_gb': kv_cache_total,
                'total_memory_gb': total_memory_per_replica,
                'required_replicas': int(required_replicas),
                'gpus_per_replica': int(gpus_per_replica),
                'required_gpus': required_gpus,
                'required_servers': required_servers,
                'hourly_cost': hourly_cost,
                'gpu_utilization': gpu_utilization,
                'memory_utilization': memory_utilization,
                'estimated_latency_ms': estimated_latency_ms,
                'model_type': model_type,
                'gpu_type': gpu_type
            })
        
        return pd.DataFrame(results)
    
    def generate_multi_scenario_data(self, 
                                   scenarios: List[Dict] = None,
                                   duration_hours: int = 24*7) -> Dict[str, pd.DataFrame]:
        """
        生成多场景仿真数据
        """
        if scenarios is None:
            scenarios = []
            
            # 定义基础配置
            model_configs = [
                {
                    'model': 'GPT-4',
                    'base_qps': {'daily': 5, 'weekly': 5, 'burst': 3, 'steady': 4},
                    'gpu_type': 'H100',  # 大模型需要H100
                    'redundancy': {'daily': 1.5, 'weekly': 1.4, 'burst': 1.8, 'steady': 1.3}
                },
                {
                    'model': 'ChatGLM',
                    'base_qps': {'daily': 300, 'weekly': 250, 'burst': 200, 'steady': 350},
                    'gpu_type': 'A100',
                    'redundancy': {'daily': 1.3, 'weekly': 1.3, 'burst': 1.6, 'steady': 1.2}
                },
                {
                    'model': 'Claude',
                    'base_qps': {'daily': 10, 'weekly': 10, 'burst': 8, 'steady': 12},
                    'gpu_type': 'H100',  # 长上下文需要H100
                    'redundancy': {'daily': 1.3, 'weekly': 1.3, 'burst': 1.5, 'steady': 1.2}
                },
                {
                    'model': 'LLaMA',
                    'base_qps': {'daily': 800, 'weekly': 700, 'burst': 500, 'steady': 900},
                    'gpu_type': 'A100',
                    'redundancy': {'daily': 1.3, 'weekly': 1.3, 'burst': 1.5, 'steady': 1.2}
                }
            ]
            
            # 为每个模型生成所有模式的场景
            patterns = ['daily', 'weekly', 'burst', 'steady']
            pattern_desc = {
                'daily': 'Daily',
                'weekly': 'Weekly', 
                'burst': 'Burst',
                'steady': 'Steady'
            }
            
            for config in model_configs:
                for pattern in patterns:
                    scenario = {
                        'name': f"{config['model']}_{pattern_desc[pattern]}",
                        'model_type': config['model'],
                        'pattern_type': pattern,
                        'base_qps': config['base_qps'][pattern],
                        'gpu_type': config['gpu_type'],
                        'redundancy': config['redundancy'][pattern]
                    }
                    scenarios.append(scenario)
        
        results = {}
        
        for scenario in scenarios:
            print(f"生成场景: {scenario['name']}")
            
            # 生成流量数据
            traffic_df = self.generate_traffic_pattern(
                duration_hours=duration_hours,
                base_qps=scenario['base_qps'],
                pattern_type=scenario['pattern_type'],
                model_type=scenario['model_type']
            )
            
            # 计算资源需求
            resource_df = self.calculate_resource_requirements(
                traffic_df=traffic_df,
                model_type=scenario['model_type'],
                gpu_type=scenario['gpu_type'],
                redundancy_factor=scenario['redundancy']
            )
            
            results[scenario['name']] = resource_df
        
        return results
    
    def save_data(self, data: Dict[str, pd.DataFrame], output_dir: str = 'simulation_data'):
        """
        保存仿真数据
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存各场景数据
        for scenario_name, df in data.items():
            filename = f"{scenario_name}.csv"
            filepath = os.path.join(output_dir, filename)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"已保存: {filepath}")
        
        # 合并所有数据
        all_data = []
        for scenario_name, df in data.items():
            df_copy = df.copy()
            df_copy['scenario'] = scenario_name
            all_data.append(df_copy)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_filepath = os.path.join(output_dir, 'combined_simulation_data.csv')
        combined_df.to_csv(combined_filepath, index=False, encoding='utf-8-sig')
        print(f"已保存合并数据: {combined_filepath}")
        
        # 保存配置信息
        config_info = {
            'model_configs': self.model_configs,
            'hardware_configs': self.hardware_configs,
            'generation_time': datetime.now().isoformat(),
            'description': '基于真实LLM部署经验的仿真数据'
        }
        
        config_filepath = os.path.join(output_dir, 'simulation_config.json')
        with open(config_filepath, 'w', encoding='utf-8') as f:
            json.dump(config_info, f, indent=2, ensure_ascii=False)
        print(f"已保存配置: {config_filepath}")
    
    def plot_analysis(self, data: Dict[str, pd.DataFrame], output_dir: str = 'simulation_data'):
        """
        生成分析图表
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. 资源需求对比图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('LLM Resource Requirements Analysis', fontsize=16, fontweight='bold')
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (scenario_name, df) in enumerate(data.items()):
            color = colors[i % len(colors)]
            
            # 采样数据以减少图表密度
            sample_rate = max(1, len(df) // 500)
            df_sample = df.iloc[::sample_rate]
            
            # QPS趋势
            axes[0, 0].plot(df_sample.index, df_sample['qps'], 
                          label=scenario_name, color=color, alpha=0.7, linewidth=1.5)
            
            # GPU需求趋势
            axes[0, 1].plot(df_sample.index, df_sample['required_gpus'], 
                          label=scenario_name, color=color, alpha=0.7, linewidth=1.5)
            
            # 服务器需求趋势
            axes[0, 2].plot(df_sample.index, df_sample['required_servers'], 
                          label=scenario_name, color=color, alpha=0.7, linewidth=1.5)
            
            # 成本趋势
            axes[1, 0].plot(df_sample.index, df_sample['hourly_cost'], 
                          label=scenario_name, color=color, alpha=0.7, linewidth=1.5)
            
            # GPU利用率
            axes[1, 1].plot(df_sample.index, df_sample['gpu_utilization'] * 100, 
                          label=scenario_name, color=color, alpha=0.7, linewidth=1.5)
            
            # 延迟估算
            axes[1, 2].plot(df_sample.index, df_sample['estimated_latency_ms'], 
                          label=scenario_name, color=color, alpha=0.7, linewidth=1.5)
        
        # 设置子图属性
        axes[0, 0].set_title('QPS Trend')
        axes[0, 0].set_ylabel('Queries per Second')
        axes[0, 0].legend(fontsize=8)
        
        axes[0, 1].set_title('GPU Requirements')
        axes[0, 1].set_ylabel('Number of GPUs')
        
        axes[0, 2].set_title('Server Requirements')
        axes[0, 2].set_ylabel('Number of Servers')
        
        axes[1, 0].set_title('Hourly Cost')
        axes[1, 0].set_ylabel('Cost (USD)')
        axes[1, 0].set_xlabel('Time (samples)')
        
        axes[1, 1].set_title('GPU Utilization')
        axes[1, 1].set_ylabel('Utilization (%)')
        axes[1, 1].set_xlabel('Time (samples)')
        
        axes[1, 2].set_title('Estimated Latency')
        axes[1, 2].set_ylabel('Latency (ms)')
        axes[1, 2].set_xlabel('Time (samples)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'resource_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 统计分布图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Resource Distribution Analysis', fontsize=16, fontweight='bold')
        
        # GPU需求分布
        for scenario_name, df in data.items():
            axes[0, 0].hist(df['required_gpus'], bins=30, alpha=0.5, label=scenario_name)
        axes[0, 0].set_title('GPU Requirements Distribution')
        axes[0, 0].set_xlabel('Number of GPUs')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # 成本分布
        for scenario_name, df in data.items():
            axes[0, 1].hist(df['hourly_cost'], bins=30, alpha=0.5, label=scenario_name)
        axes[0, 1].set_title('Hourly Cost Distribution')
        axes[0, 1].set_xlabel('Cost (USD)')
        axes[0, 1].set_ylabel('Frequency')
        
        # 利用率箱线图
        util_data = []
        scenario_names = []
        for scenario_name, df in data.items():
            util_data.extend(df['gpu_utilization'].tolist())
            scenario_names.extend([scenario_name] * len(df))
        
        util_df = pd.DataFrame({'utilization': util_data, 'scenario': scenario_names})
        sns.boxplot(data=util_df, x='scenario', y='utilization', ax=axes[1, 0])
        axes[1, 0].set_title('GPU Utilization Distribution')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # QPS vs GPU scatter plot
        for scenario_name, df in data.items():
            # 采样以避免过多点
            sample_indices = np.random.choice(len(df), min(1000, len(df)), replace=False)
            axes[1, 1].scatter(df.iloc[sample_indices]['qps'], 
                             df.iloc[sample_indices]['required_gpus'],
                             alpha=0.5, s=20, label=scenario_name)
        axes[1, 1].set_title('QPS vs GPU Requirements')
        axes[1, 1].set_xlabel('QPS')
        axes[1, 1].set_ylabel('Required GPUs')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'resource_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"图表已保存到: {output_dir}")


def main():
    """
    主函数：运行完整的仿真流程
    """
    print("=" * 60)
    print("大模型算力资源调度仿真数据生成器 - 改进版")
    print("基于真实LLM部署经验的参数配置")
    print("=" * 60)
    
    # 创建仿真器
    simulator = ResourceSimulator(seed=42)
    
    # 生成多场景数据
    print("\n开始生成仿真数据...")
    simulation_data = simulator.generate_multi_scenario_data(duration_hours=24*7)  # 一周数据
    
    # 保存数据
    print("\n保存仿真数据...")
    simulator.save_data(simulation_data)
    
    # 生成分析图表
    print("\n生成分析图表...")
    simulator.plot_analysis(simulation_data)
    
    # 输出统计信息
    print("\n=== 仿真数据统计 ===")
    for scenario_name, df in simulation_data.items():
        print(f"\n场景: {scenario_name}")
        print(f"  数据点数: {len(df)}")
        print(f"  QPS范围: [{df['qps'].min():.1f}, {df['qps'].max():.1f}]")
        print(f"  平均QPS: {df['qps'].mean():.2f}")
        print(f"  GPU需求范围: [{df['required_gpus'].min()}, {df['required_gpus'].max()}]")
        print(f"  平均GPU需求: {df['required_gpus'].mean():.1f}")
        print(f"  平均每小时成本: ${df['hourly_cost'].mean():.2f}")
        print(f"  平均GPU利用率: {df['gpu_utilization'].mean():.1%}")
        print(f"  平均延迟: {df['estimated_latency_ms'].mean():.1f}ms")
    
    print("\n仿真数据生成完成！")
    print("数据文件保存在 'simulation_data' 目录中")
    print("\n改进说明：")
    print("1. 基于真实的LLM部署经验调整了参数")
    print("2. 考虑了prefill和generation阶段的不同计算需求")
    print("3. 加入了KV cache内存计算")
    print("4. 使用了更真实的GPU吞吐量（2000-3000 tokens/s）")
    print("5. 考虑了模型并行的最小GPU需求")
    print("6. 添加了延迟估算和更多运营指标")


if __name__ == "__main__":
    main()