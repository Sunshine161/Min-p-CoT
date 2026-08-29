import torch
import torch.nn.functional as F
import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Union, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
from extractor import extract_answer

# --- 辅助函数: 日志设置 ---
def setup_logging(output_dir: str, experiment_name: str = None) -> logging.Logger:
    """
    设置日志系统，记录所有状态信息
    
    Args:
        output_dir: 输出目录
        experiment_name: 实验名称，如果不提供会自动生成
    
    Returns:
        配置好的logger对象
    """
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 生成实验名称
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"dynamics_minp_{timestamp}"
    
    # 设置日志文件路径到log文件夹
    log_dir = Path(output_dir) / "log"
    log_dir.mkdir(exist_ok=True)  # 确保log文件夹存在
    log_file = log_dir / f"{experiment_name}.log"
    
    # 创建logger
    logger = logging.getLogger("dynamics_minp")
    logger.setLevel(logging.INFO)
    
    # 清除已有的handlers
    logger.handlers.clear()
    
    # 创建文件handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 设置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"日志系统初始化完成，日志文件: {log_file}")
    return logger

# --- 辅助函数: 配置文件加载 ---
def load_config(config_path: str = "/cephfs/shared/sunyifan/Min-p-CoT/dynamics_minp_config.json") -> Dict:
    """
    从JSON配置文件加载所有超参数。
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        Dict: 配置参数字典
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"成功加载配置文件: {config_path}")
        return config
    except json.JSONDecodeError as e:
        raise ValueError(f"配置文件格式错误: {e}")
    except Exception as e:
        raise Exception(f"加载配置文件失败: {e}")


def get_torch_dtype(dtype_str: str):
    """
    将字符串转换为torch数据类型。
    
    Args:
        dtype_str: 数据类型字符串
        
    Returns:
        torch.dtype: 对应的torch数据类型
    """
    dtype_mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float": torch.float,
        "half": torch.half
    }
    
    return dtype_mapping.get(dtype_str, torch.bfloat16)


# --- 辅助函数: 本地模型和数据集加载 ---
def load_local_dataset(dataset_path: Union[str, Path]) -> List[Dict]:
    """
    从本地文件加载数据集，支持带有标准答案的数据格式。
    
    Args:
        dataset_path: 数据集文件路径，支持 .txt, .json, .jsonl 格式
        
    Returns:
        List[Dict]: 包含问题和答案的数据列表，格式为 [{"question": str, "answer": str, "dataset_type": str}, ...]
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")
    
    dataset_items = []
    
    if dataset_path.suffix == '.txt':
        # 简单文本格式，每行一个问题，没有标准答案
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    dataset_items.append({
                        "question": line,
                        "answer": None,
                        "dataset_type": "math"
                    })
    
    elif dataset_path.suffix == '.json':
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        dataset_items.append({
                            "question": item,
                            "answer": None,
                            "dataset_type": "math"
                        })
                    elif isinstance(item, dict):
                        # 支持多种数据格式
                        question = item.get('question', item.get('problem', item.get('text', '')))
                        
                        # 尝试从多个字段获取答案
                        answer = item.get('answer', item.get('ground_truth', None))
                        
                        # 如果没有直接的答案，尝试从solution中提取
                        if answer is None and 'solution' in item:
                            try:
                                # 使用extractor从solution中提取答案
                                answer = extract_answer(item['solution'], 'math', reference_answer=True)
                            except:
                                answer = None
                        
                        dataset_items.append({
                            "question": question,
                            "answer": answer,
                            "dataset_type": item.get('dataset_type', 'math')
                        })
            elif isinstance(data, dict):
                dataset_items.append({
                    "question": data.get('question', data.get('text', '')),
                    "answer": data.get('answer', data.get('ground_truth', None)),
                    "dataset_type": data.get('type', data.get('dataset_type', 'math'))
                })
    
    elif dataset_path.suffix == '.jsonl':
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    if isinstance(item, str):
                        dataset_items.append({
                            "question": item,
                            "answer": None,
                            "dataset_type": "math"
                        })
                    elif isinstance(item, dict):
                        # 支持多种数据格式
                        question = item.get('question', item.get('problem', item.get('text', '')))
                        
                        # 尝试从多个字段获取答案
                        answer = item.get('answer', item.get('ground_truth', None))
                        
                        # 如果没有直接的答案，尝试从solution中提取
                        if answer is None and 'solution' in item:
                            try:
                                # 使用extractor从solution中提取答案
                                answer = extract_answer(item['solution'], 'math', reference_answer=True)
                            except:
                                answer = None
                        
                        dataset_items.append({
                            "question": question,
                            "answer": answer,
                            "dataset_type": item.get('dataset_type', 'math'),
                            "level": item.get('level', ''),
                            "category": item.get('category', item.get('type', ''))
                        })
    
    else:
        raise ValueError(f"不支持的文件格式: {dataset_path.suffix}。支持的格式: .txt, .json, .jsonl")
    
    return [item for item in dataset_items if item["question"]]


# --- 辅助函数: 多阶段Prompt生成 ---
def get_stage_prompts(config: dict, question: str, dataset_type: str = "math") -> List[str]:
    """
    从配置文件生成4阶段CoT的提示列表
    
    Args:
        config: 配置字典
        question: 原始问题
        dataset_type: 数据集类型
    
    Returns:
        List[str]: 4个阶段的提示列表
    """
    cot_prompts = config.get('cot_prompts', {})
    stage_config = cot_prompts.get('4_stage', {}).get(dataset_type.lower(), {})
    
    if dataset_type.lower() == "math":
        stage_prompts = [
            # 阶段1: 问题理解
            f"{question}\n\n{stage_config.get('stage1', 'First, let me understand this problem. What information is given and what needs to be found?')}",
            
            # 阶段2: 策略规划
            stage_config.get('stage2', "Now, let me plan my approach. What method or formula should I use to solve this problem?"),
            
            # 阶段3: 解题执行
            stage_config.get('stage3', "Let me work through the solution step by step with calculations:"),
            
            # 阶段4: 验证和最终答案
            stage_config.get('stage4', "Finally, let me verify my solution and present the final answer in \\boxed{{}} format:")
        ]
    else:
        stage_prompts = [
            # 阶段1: 理解问题
            f"{question}\n\n{stage_config.get('stage1', 'First, let me understand what this question is asking:')}",
            
            # 阶段2: 分析要点
            stage_config.get('stage2', "Now, let me analyze the key points and relevant information:"),
            
            # 阶段3: 推理过程
            stage_config.get('stage3', "Let me work through the reasoning step by step:"),
            
            # 阶段4: 总结结论
            stage_config.get('stage4', "Finally, let me provide a clear and well-supported conclusion:")
        ]
    
    return stage_prompts

def enhance_prompt_with_cot(config: dict, question: str, dataset_type: str = "math", cot_method: str = "simple") -> str:
    """
    根据配置为问题添加思维链提示
    
    Args:
        config: 配置字典
        question: 原始问题
        dataset_type: 数据集类型
        cot_method: CoT方法类型
    
    Returns:
        增强后的prompt
    """
    cot_prompts = config.get('cot_prompts', {})
    
    if cot_method == "4_stage" or cot_method == "4_stage_cot":
        # 4阶段CoT在multi_stage_generate中处理，这里返回原始问题
        return question
    elif cot_method == "none":
        # 不使用CoT
        return question
    else:
        # 简单CoT（默认）
        return enhance_simple_cot(cot_prompts, question, dataset_type)

def enhance_simple_cot(cot_prompts: dict, question: str, dataset_type: str) -> str:
    """
    添加简单CoT提示
    """
    simple_config = cot_prompts.get('simple', {})
    
    if dataset_type.lower() == "math":
        prompt_suffix = simple_config.get('math', "Let's think step by step and provide the final answer in \\boxed{{}} format.")
    else:
        prompt_suffix = simple_config.get('qa', "Let's think step by step.")
    
    return f"{question}\n\n{prompt_suffix}"

def verify_local_model(model_path: Union[str, Path]) -> bool:
    """
    验证本地模型路径是否有效。
    
    Args:
        model_path: 模型文件夹路径
        
    Returns:
        bool: 路径是否有效
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        return False
    
    # 检查必要的模型文件
    required_files = ['config.json']
    optional_files = ['pytorch_model.bin', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json']
    
    has_required = all((model_path / file).exists() for file in required_files)
    has_model_weights = any((model_path / file).exists() for file in optional_files)
    
    return has_required and has_model_weights


# --- 核心组件 1: 熵计算 ---
# 这是一个辅助函数，用于从概率分布中计算熵
def calculate_entropy(probs: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
    """
    计算给定概率分布张量的熵。
    
    Args:
        probs (torch.Tensor): 模型的输出概率分布，形状为 (batch_size, vocab_size)。
        epsilon (float): 一个很小的数，用于防止 log(0) 导致NaN。

    Returns:
        torch.Tensor: 每个批次样本的熵值，形状为 (batch_size,)。
    """
    # 添加 epsilon 以保证数值稳定性
    probs_stable = probs + epsilon
    entropy = -torch.sum(probs * torch.log(probs_stable), dim=-1)
    return entropy


# --- 核心组件 2: 动态 min_p LogitsProcessor ---
# 这是我们整个策略的核心，它被封装成一个 LogitsProcessor 类
class EntropyDynamicMinPLogitsProcessor(LogitsProcessor):
    """
    一个 LogitsProcessor，它根据模型输出分布的熵来动态调整 min_p 的值。
    - 高熵 (不确定) -> 使用线性映射的 min_p (更严格的过滤)
    - 低熵 (确定)   -> 使用固定的 min_p = 0.2
    - 支持滑动窗口动态估计熵的最大值
    """
    def __init__(self, h_min: float, h_max: float, min_p_min: float, min_p_max: float, 
                 use_sliding_window: bool = False, window_size: int = 10, 
                 percentile: float = 95, min_samples_for_update: int = 5,
                 entropy_threshold: float = None, fixed_low_minp: float = 0.2):
        """
        初始化动态 min_p 控制器。

        Args:
            h_min (float): 熵的经验最小值 (通常设为 0.0)。
            h_max (float): 熵的初始最大值或固定最大值。
            min_p_min (float): 目标 min_p 范围的最小值（高熵时的线性映射范围）。
            min_p_max (float): 目标 min_p 范围的最大值（高熵时的线性映射范围）。
            use_sliding_window (bool): 是否使用滑动窗口动态估计h_max。
            window_size (int): 滑动窗口大小。
            percentile (float): 用于计算动态h_max的百分位数。
            min_samples_for_update (int): 更新h_max所需的最小样本数。
            entropy_threshold (float): 熵阈值，低于此值使用固定min_p，高于此值使用线性映射。如果为None，则使用h_max的一半。
            fixed_low_minp (float): 低熵时使用的固定min_p值，默认为0.2。
        """
        self.h_min = h_min
        self.h_max = h_max
        self.initial_h_max = h_max
        self.min_p_min = min_p_min
        self.min_p_max = min_p_max
        
        # 新增：熵阈值和固定min_p值
        self.entropy_threshold = entropy_threshold if entropy_threshold is not None else h_max / 2
        self.fixed_low_minp = fixed_low_minp
        
        # 滑动窗口相关参数
        self.use_sliding_window = use_sliding_window
        self.window_size = window_size
        self.percentile = percentile
        self.min_samples_for_update = min_samples_for_update
        
        # 熵值记录
        self.entropy_history = []  # 记录每个token的熵值
        self.batch_entropy_histories = {}  # 记录每个批次样本的熵值历史
        self.entropy_window = []  # 滑动窗口用于动态h_max计算
        self.dynamic_h_max_history = []  # 记录动态h_max的变化历史
        
        # min_p值记录
        self.minp_history = []  # 记录每个token实际使用的min_p值
        self.batch_minp_histories = {}  # 记录每个批次样本的min_p值历史

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 1. 从 logits 计算概率和熵
        # scores 的形状是 (batch_size, vocab_size)
        probs = F.softmax(scores, dim=-1)
        entropy = calculate_entropy(probs) # 熵的形状是 (batch_size,)

        # 记录熵值（支持批量记录）
        current_entropy = None
        if entropy.numel() > 0:
            current_entropy = entropy[0].item()
            
            # 记录第一个样本的熵值（向后兼容）
            self.entropy_history.append(current_entropy)
            
            # 更新滑动窗口
            if self.use_sliding_window:
                self.entropy_window.append(current_entropy)
                # 保持窗口大小
                if len(self.entropy_window) > self.window_size:
                    self.entropy_window.pop(0)
            
            # 记录所有批次样本的熵值
            batch_size = entropy.size(0)
            for batch_idx in range(batch_size):
                if batch_idx not in self.batch_entropy_histories:
                    self.batch_entropy_histories[batch_idx] = []
                self.batch_entropy_histories[batch_idx].append(entropy[batch_idx].item())

        # 2. 动态更新h_max（如果使用滑动窗口）
        current_h_max = self.h_max
        if self.use_sliding_window and len(self.entropy_window) >= self.min_samples_for_update:
            # 使用百分位数计算动态h_max
            import numpy as np
            dynamic_h_max = np.percentile(self.entropy_window, self.percentile)
            
            # 确保动态h_max不会太小，至少是初始值的一定比例
            min_allowed_h_max = self.initial_h_max * 0.3
            dynamic_h_max = max(dynamic_h_max, min_allowed_h_max)
            
            current_h_max = dynamic_h_max
            self.dynamic_h_max_history.append(dynamic_h_max)
            
            # 可选：平滑更新h_max以避免剧烈波动
            alpha = 0.1  # 平滑因子
            self.h_max = alpha * dynamic_h_max + (1 - alpha) * self.h_max

        # 3. 根据熵值决定使用哪种策略
        # 创建一个与entropy相同形状的tensor来存储dynamic_min_p
        batch_size = entropy.size(0)
        dynamic_min_p = torch.zeros(batch_size, 1).to(entropy.device)
        
        # 判断每个样本的熵是否超过阈值
        # 使用动态阈值（如果启用滑动窗口）
        current_threshold = self.entropy_threshold
        if self.use_sliding_window and len(self.entropy_window) >= self.min_samples_for_update:
            # 动态调整阈值为当前h_max的一定比例
            current_threshold = current_h_max * 0.4  # 使用40%作为阈值
        
        # 创建掩码：True表示高熵（使用线性映射），False表示低熵（使用固定值）
        high_entropy_mask = entropy > current_threshold
        
        # 4. 对高熵样本使用线性映射，对低熵样本使用固定值
        if high_entropy_mask.any():
            # 计算高熵样本的线性映射
            # 只对高于阈值的部分进行线性映射
            entropy_ratio = (entropy - current_threshold) / (current_h_max - current_threshold)
            entropy_ratio = torch.clamp(entropy_ratio, 0.0, 1.0)
            
            # 线性映射到 [min_p_min, min_p_max] 范围
            linear_min_p = self.min_p_min + entropy_ratio * (self.min_p_max - self.min_p_min)
            
            # 应用到高熵样本
            dynamic_min_p[high_entropy_mask] = linear_min_p[high_entropy_mask].unsqueeze(-1)
        
        # 对低熵样本使用固定的min_p值
        low_entropy_mask = ~high_entropy_mask
        if low_entropy_mask.any():
            dynamic_min_p[low_entropy_mask] = self.fixed_low_minp
        
        # 记录每个token的实际min_p值
        current_minp = None
        if dynamic_min_p.numel() > 0:
            current_minp = dynamic_min_p[0].item()
            
            # 记录第一个样本的min_p值（向后兼容）
            self.minp_history.append(current_minp)
            
            # 记录所有批次样本的min_p值
            batch_size = dynamic_min_p.size(0)
            for batch_idx in range(batch_size):
                if batch_idx not in self.batch_minp_histories:
                    self.batch_minp_histories[batch_idx] = []
                self.batch_minp_histories[batch_idx].append(dynamic_min_p[batch_idx].item())
        
        # 5. 实现并应用 min_p 过滤逻辑
        # 找到每个批次中的最大概率 P_max
        p_max, _ = torch.max(probs, dim=-1, keepdim=True)
        
        # 计算过滤阈值：threshold = dynamic_min_p * P_max
        threshold = dynamic_min_p * p_max
        
        # 找到所有概率低于阈值的 token
        indices_to_remove = probs < threshold
        
        # 将这些 token 的 logits 设置为负无穷，使其在采样中被忽略
        scores[indices_to_remove] = -float("inf")

        # 打印调试信息 (可选)
        if self.use_sliding_window and current_entropy is not None and current_minp is not None:
            strategy = "Linear" if current_entropy > current_threshold else "Fixed"
            debug_info = f"Entropy: {current_entropy:.4f}, "
            debug_info += f"Threshold: {current_threshold:.4f}, "
            debug_info += f"Strategy: {strategy}, "
            debug_info += f"Current h_max: {current_h_max:.4f}, "
            debug_info += f"Dynamic min_p: {current_minp:.4f}, "
            debug_info += f"Window size: {len(self.entropy_window)}"
            # print(debug_info)  # 取消注释以启用调试输出

        return scores
    
    def reset_entropy_history(self):
        """重置熵值和min_p记录"""
        self.entropy_history = []
        self.batch_entropy_histories = {}
        self.minp_history = []
        self.batch_minp_histories = {}
        if self.use_sliding_window:
            self.entropy_window = []
            self.dynamic_h_max_history = []
            self.h_max = self.initial_h_max  # 重置为初始值
    
    def get_entropy_history(self) -> List[float]:
        """获取熵值记录（向后兼容，返回第一个样本的熵值）"""
        return self.entropy_history.copy()
    
    def get_batch_entropy_histories(self) -> Dict[int, List[float]]:
        """获取所有批次样本的熵值历史"""
        return {k: v.copy() for k, v in self.batch_entropy_histories.items()}
    
    def get_entropy_history_for_batch(self, batch_idx: int) -> List[float]:
        """获取指定批次样本的熵值历史"""
        return self.batch_entropy_histories.get(batch_idx, []).copy()
    
    def get_minp_history(self) -> List[float]:
        """获取min_p值记录（向后兼容，返回第一个样本的min_p值）"""
        return self.minp_history.copy()
    
    def get_batch_minp_histories(self) -> Dict[int, List[float]]:
        """获取所有批次样本的min_p值历史"""
        return {k: v.copy() for k, v in self.batch_minp_histories.items()}
    
    def get_minp_history_for_batch(self, batch_idx: int) -> List[float]:
        """获取指定批次样本的min_p值历史"""
        return self.batch_minp_histories.get(batch_idx, []).copy()
    
    def get_sliding_window_stats(self) -> Dict:
        """获取滑动窗口统计信息"""
        if not self.use_sliding_window:
            return {"enabled": False}
        
        stats = {
            "enabled": True,
            "window_size": self.window_size,
            "current_window_size": len(self.entropy_window),
            "percentile": self.percentile,
            "current_h_max": self.h_max,
            "initial_h_max": self.initial_h_max,
            "dynamic_h_max_history": self.dynamic_h_max_history.copy()
        }
        
        if self.entropy_window:
            import numpy as np
            stats.update({
                "window_mean": np.mean(self.entropy_window),
                "window_std": np.std(self.entropy_window),
                "window_min": np.min(self.entropy_window),
                "window_max": np.max(self.entropy_window)
            })
        
        return stats


def run_multiple_trials(model, tokenizer, questions: List[str], dataset_items: List[Dict], dataset_type: str, dynamic_min_p_processor, generation_config: Dict, config: Dict, logger, num_trials: int = 1) -> Dict:
    """
    运行多次试验并计算平均结果
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        questions: 问题列表
        dataset_items: 数据集项目列表
        dataset_type: 数据集类型
        dynamic_min_p_processor: 动态min_p处理器
        generation_config: 生成配置
        config: 完整配置
        logger: 日志记录器
        num_trials: 试验次数
    
    Returns:
        Dict: 包含所有试验结果和平均值的字典
    """
    logger.info(f"开始进行 {num_trials} 次试验")
    
    all_trials_results = []
    cot_method = config.get('generation', {}).get('cot_method', 'simple')
    enable_batch = config.get('generation', {}).get('enable_batch_generation', False)
    batch_size = config.get('generation', {}).get('batch_size', 4)
    use_one_shot = config.get('one_shot', False)
    one_shot_prompts = config.get('one_shot_prompts', '')
    
    for trial_idx in range(num_trials):
        logger.info(f"\n=== 试验 {trial_idx + 1}/{num_trials} ===")
        
        trial_results = {
            'trial_id': trial_idx + 1,
            'responses': [],
            'entropy_histories': [],
            'minp_histories': [],
            'predicted_answers': [],
            'ground_truth_answers': [],
            'generation_times': [],
            'stage_responses': [],  # 用于4阶段CoT
            'stage_entropy_histories': [],
            'stage_minp_histories': []
        }
        
        # 重置处理器的滑动窗口状态
        dynamic_min_p_processor.reset_entropy_history()
        
        # 读取投票配置
        voting_config = config.get('voting', {})
        enable_voting = voting_config.get('enable_voting', False)
        num_votes = voting_config.get('num_votes', 5)
        voting_temperature = voting_config.get('voting_temperature', 0.8)
        voting_strategy = voting_config.get('voting_strategy', 'majority')
        use_parallel_voting = voting_config.get('use_parallel_voting', True)
        confidence_threshold = voting_config.get('confidence_threshold', 0.0)  # 置信度阈值
        enable_greedy_fallback = voting_config.get('enable_greedy_fallback', True)  # 是否启用贪婪回退
        
        logger.info(f"CoT方法: {cot_method}, 批量生成: {enable_batch}")
        if enable_voting:
            parallel_mode = "并行" if use_parallel_voting else "串行"
            logger.info(f"启用投票机制: {num_votes}次{parallel_mode}投票, 温度: {voting_temperature}, 策略: {voting_strategy}")
            if confidence_threshold > 0:
                logger.info(f"置信度阈值: {confidence_threshold:.2f}, 贪婪回退: {'启用' if enable_greedy_fallback else '禁用'}")
        
        # 统一的投票生成逻辑
        if enable_voting:
            logger.info(f"进入{parallel_mode}投票生成模式，CoT方法: {cot_method}")
            start_time = time.time()
            
            # 检查是否启用批量并行投票
            use_batch_voting = enable_batch and use_parallel_voting
            
            if use_batch_voting:
                logger.info(f"🚀 启用批量并行投票模式，批量大小: {batch_size}")
                
                # 构建问题列表
                questions = [item["question"] for item in dataset_items]
                
                # 批量并行投票生成
                all_question_responses, all_question_extra_info = generate_batch_voting_responses(
                    model, tokenizer, questions, dataset_type, dynamic_min_p_processor,
                    generation_config, config, num_votes, voting_temperature, cot_method, batch_size
                )
                
                # 处理批量投票结果
                for i, item in enumerate(dataset_items):
                    if i < len(all_question_responses):
                        combined_responses = all_question_responses[i]
                        extra_info = all_question_extra_info[i] if i < len(all_question_extra_info) else []
                        
                        # 提取所有响应的Answer内容
                        full_responses = [extract_answer_content(resp) for resp in combined_responses]
                        
                        # 进行投票
                        final_answer, voting_details = majority_vote(full_responses, dataset_type, confidence_threshold)
                        
                        # 检查置信度是否满足阈值
                        if not voting_details.get('meets_threshold', True) and enable_greedy_fallback:
                            # 使用贪婪解码回退
                            original_question = dataset_items[i]["question"]
                            enhanced_question = enhance_prompt_with_cot(config, original_question, dataset_type, cot_method)
                            greedy_answer = generate_greedy_fallback(
                                model, tokenizer, enhanced_question, dynamic_min_p_processor, 
                                generation_config, dataset_type
                            )
                            predicted_answer = greedy_answer if greedy_answer else final_answer
                            voting_details['used_greedy_fallback'] = True
                            voting_details['greedy_answer'] = greedy_answer
                        else:
                            predicted_answer = final_answer
                            voting_details['used_greedy_fallback'] = False
                        
                        # 使用第一个响应作为代表进行记录
                        combined_response = combined_responses[0] if combined_responses else ""
                        full_response = full_responses[0] if full_responses else ""
                        
                        # 处理4阶段CoT的特殊信息
                        if cot_method == "4_stage_cot" and extra_info:
                            all_stage_responses = extra_info
                            # 找到获胜答案对应的候选索引
                            winner_idx = 0
                            for idx, resp in enumerate(full_responses):
                                if extract_answer(resp, dataset_type) == predicted_answer:
                                    winner_idx = idx
                                    break
                            stage_responses = all_stage_responses[winner_idx] if winner_idx < len(all_stage_responses) else []
                            # 为了兼容原有格式，创建空的熵值和min_p历史
                            stage_entropy_histories = [[] for _ in range(4)]
                            stage_minp_histories = [[] for _ in range(4)]
                        else:
                            # 非4阶段CoT的情况
                            stage_responses = []
                            stage_entropy_histories = []
                            stage_minp_histories = []
                        
                        # 判断正确性
                        is_correct = str(predicted_answer).strip().lower() == str(item.get('answer', '')).strip().lower()
                        
                        # 合并熵值和min_p历史
                        combined_entropy_history = []
                        combined_minp_history = []
                        for stage_entropy, stage_minp in zip(stage_entropy_histories, stage_minp_histories):
                            combined_entropy_history.extend(stage_entropy)
                            combined_minp_history.extend(stage_minp)
                        
                        # 记录所有样本的结果（无论正确还是错误）
                        trial_results['responses'].append(full_response)
                        trial_results['entropy_histories'].append(combined_entropy_history)
                        trial_results['minp_histories'].append(combined_minp_history)
                        trial_results['stage_responses'].append(stage_responses)
                        trial_results['stage_entropy_histories'].append(stage_entropy_histories)
                        trial_results['stage_minp_histories'].append(stage_minp_histories)
                        trial_results['generation_times'].append((time.time() - start_time) / (i + 1))
                        trial_results['predicted_answers'].append(predicted_answer)
                        trial_results['ground_truth_answers'].append(item.get("answer", ""))
                        
                        # 只打印预测错误的样本
                        if not is_correct:
                            logger.info(f"\n--- ❌ 预测错误样本 {i + 1}/{len(dataset_items)} ---")
                            logger.info(f"问题: {item['question']}")
                            logger.info(f"标准答案: {item.get('answer', 'N/A')}")
                            logger.info(f"预测答案: {predicted_answer}")
                            logger.info(f"投票详情: {voting_details}")
                            
                            # 显示置信度和回退信息
                            if voting_details.get('used_greedy_fallback', False):
                                logger.info(f"⚠️  投票置信度 {voting_details['confidence']:.3f} 低于阈值 {voting_details['threshold']:.3f}")
                                logger.info(f"🔄 使用贪婪解码回退，答案: {voting_details.get('greedy_answer', 'N/A')}")
                            
                            logger.info("所有候选响应:")
                            for j, resp in enumerate(full_responses):
                                candidate_answer = extract_answer(resp, dataset_type)
                                logger.info(f"  候选{j+1}: {candidate_answer}")
                            
                            logger.info(f"🔗 获胜完整推理链: {combined_response}")
                            logger.info(f"📝 最终提取答案: {full_response}")
                    else:
                        # 处理缺失的结果
                        trial_results['responses'].append("")
                        trial_results['entropy_histories'].append([])
                        trial_results['minp_histories'].append([])
                        trial_results['stage_responses'].append([])
                        trial_results['stage_entropy_histories'].append([])
                        trial_results['stage_minp_histories'].append([])
                        trial_results['predicted_answers'].append("")
                        trial_results['ground_truth_answers'].append(item.get("answer", ""))
                        trial_results['generation_times'].append(0)
            
            else:
                # 原有的逐个投票处理逻辑
                logger.info(f"使用逐个投票处理模式")
                for i, item in enumerate(dataset_items):
                    question = item["question"]
                    
                    # 使用统一的多响应生成函数
                    combined_responses, extra_info = generate_multiple_responses_universal(
                        model, tokenizer, question, dataset_type, dynamic_min_p_processor,
                        generation_config, config, num_votes, voting_temperature, cot_method, use_parallel_voting,
                        question_idx=i, total_questions=len(dataset_items)
                    )
                    
                    # 提取所有响应的Answer内容
                    full_responses = [extract_answer_content(resp) for resp in combined_responses]
                    
                    # 进行投票
                    final_answer, voting_details = majority_vote(full_responses, dataset_type, confidence_threshold)
                    
                    # 检查置信度是否满足阈值
                    if not voting_details.get('meets_threshold', True) and enable_greedy_fallback:
                        # 使用贪婪解码回退
                        enhanced_question = enhance_prompt_with_cot(config, question, dataset_type, cot_method)
                        greedy_answer = generate_greedy_fallback(
                            model, tokenizer, enhanced_question, dynamic_min_p_processor, 
                            generation_config, dataset_type
                        )
                        predicted_answer = greedy_answer if greedy_answer else final_answer
                        voting_details['used_greedy_fallback'] = True
                        voting_details['greedy_answer'] = greedy_answer
                    else:
                        predicted_answer = final_answer
                        voting_details['used_greedy_fallback'] = False
                            
                    # 使用第一个响应作为代表进行记录
                    combined_response = combined_responses[0]
                    full_response = full_responses[0]
                    
                    # 处理4阶段CoT的特殊信息
                    if cot_method == "4_stage_cot" and extra_info:
                        all_stage_responses = extra_info
                        # 找到获胜答案对应的候选索引
                        winner_idx = 0
                        for idx, resp in enumerate(full_responses):
                            if extract_answer(resp, dataset_type) == predicted_answer:
                                winner_idx = idx
                                break
                        stage_responses = all_stage_responses[winner_idx]
                        # 为了兼容原有格式，创建空的熵值和min_p历史
                        stage_entropy_histories = [[] for _ in range(4)]
                        stage_minp_histories = [[] for _ in range(4)]
                    else:
                        # 非4阶段CoT的情况
                        stage_responses = []
                        stage_entropy_histories = []
                        stage_minp_histories = []
                    
                    # 判断正确性
                    is_correct = str(predicted_answer).strip().lower() == str(item.get('answer', '')).strip().lower()
                    
                    # 合并熵值和min_p历史
                    combined_entropy_history = []
                    combined_minp_history = []
                    for stage_entropy, stage_minp in zip(stage_entropy_histories, stage_minp_histories):
                        combined_entropy_history.extend(stage_entropy)
                        combined_minp_history.extend(stage_minp)
                    
                    # 记录所有样本的结果（无论正确还是错误）
                    trial_results['responses'].append(full_response)
                    trial_results['entropy_histories'].append(combined_entropy_history)
                    trial_results['minp_histories'].append(combined_minp_history)
                    trial_results['stage_responses'].append(stage_responses)
                    trial_results['stage_entropy_histories'].append(stage_entropy_histories)
                    trial_results['stage_minp_histories'].append(stage_minp_histories)
                    trial_results['generation_times'].append((time.time() - start_time) / (i + 1))
                    trial_results['predicted_answers'].append(predicted_answer)
                    trial_results['ground_truth_answers'].append(item.get("answer", ""))
                    
                    # 只打印预测错误的样本
                    if not is_correct:
                        logger.info(f"\n--- ❌ 预测错误样本 {i + 1}/{len(dataset_items)} ---")
                        logger.info(f"问题: {item['question']}")
                        logger.info(f"标准答案: {item.get('answer', 'N/A')}")
                        logger.info(f"预测答案: {predicted_answer}")
                        logger.info(f"投票详情: {voting_details}")
                        
                        # 显示置信度和回退信息
                        if voting_details.get('used_greedy_fallback', False):
                            logger.info(f"⚠️  投票置信度 {voting_details['confidence']:.3f} 低于阈值 {voting_details['threshold']:.3f}")
                            logger.info(f"🔄 使用贪婪解码回退，答案: {voting_details.get('greedy_answer', 'N/A')}")
                        
                        logger.info("所有候选响应:")
                        for j, resp in enumerate(full_responses):
                            candidate_answer = extract_answer(resp, dataset_type)
                            logger.info(f"  候选{j+1}: {candidate_answer}")
                        
                        # 打印所有候选路径的完整回答
                        logger.info("\n🔗 所有候选路径的完整推理过程:")
                        for j, combined_resp in enumerate(combined_responses):
                            logger.info(f"\n  📍 候选{j+1}完整推理链:")
                            logger.info(f"  {combined_resp}")
                        
                        # 如果是4阶段CoT，打印分阶段详情
                        if cot_method == "4_stage_cot" and extra_info:
                            logger.info("\n🔗 所有候选路径的分阶段详情:")
                            for j, stage_resps in enumerate(extra_info):
                                logger.info(f"\n  📍 候选{j+1}分阶段详情:")
                                for stage_idx, stage_resp in enumerate(stage_resps):
                                    if stage_resp.strip():
                                        logger.info(f"    阶段{stage_idx+1}: {stage_resp}")
                                    else:
                                        logger.info(f"    阶段{stage_idx+1}: [空响应]")
                            
                            # 打印获胜路径的4阶段详细信息
                            logger.info("\n🔍 获胜推理路径的4阶段分解:")
                            for stage_idx, stage_resp in enumerate(stage_responses):
                                if stage_resp.strip():
                                    logger.info(f"  📍 阶段{stage_idx+1}: {stage_resp}")
                                else:
                                    logger.info(f"  📍 阶段{stage_idx+1}: [空响应]")
                        
                        logger.info(f"🔗 获胜完整推理链: {combined_response}")
                        logger.info(f"📝 最终提取答案: {full_response}")
            
            total_time = time.time() - start_time
            
            # 计算投票模式的准确率和统计信息
            if trial_results['predicted_answers'] and trial_results['ground_truth_answers']:
                accuracy, correct_predictions = calculate_accuracy(
                    trial_results['predicted_answers'], 
                    trial_results['ground_truth_answers']
                )
                trial_results['accuracy'] = accuracy
                trial_results['correct_predictions'] = correct_predictions
                
                # 统计正确和错误的样本数量
                correct_count = sum(correct_predictions)
                total_count = len(correct_predictions)
                
                logger.info(f"\n=== 试验 {trial_idx + 1} 投票模式结果统计 ===")
                logger.info(f"总样本数: {total_count}")
                logger.info(f"正确数量: {correct_count}")
                logger.info(f"错误数量: {total_count - correct_count}")
                logger.info(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
                logger.info(f"平均生成时间: {sum(trial_results['generation_times'])/len(trial_results['generation_times']):.2f}秒/样本")
                logger.info(f"总生成时间: {total_time:.2f}秒")
            else:
                trial_results['accuracy'] = 0.0
                trial_results['correct_predictions'] = []
                logger.warning(f"试验 {trial_idx + 1} 投票模式缺少预测答案或标准答案，无法计算准确率")
                    
        else:
            # 非投票模式的完整生成逻辑
            logger.info(f"进入非投票生成模式，CoT方法: {cot_method}")
            start_time = time.time()
            
            if cot_method == "4_stage_cot":
                # 4阶段CoT生成
                if enable_batch:
                    logger.info(f"使用批量4阶段CoT生成，批量大小: {batch_size}")
                    all_stage_responses, all_stage_entropy_histories, all_stage_minp_histories = batch_multi_stage_generate(
                        model, tokenizer, questions, dataset_type, dynamic_min_p_processor, 
                        generation_config, config, batch_size, use_one_shot, one_shot_prompts
                    )
                    
                    # 处理4阶段结果
                    for i, item in enumerate(dataset_items):
                        if i < len(all_stage_responses):
                            stage_responses = all_stage_responses[i]
                            stage_entropy_histories = all_stage_entropy_histories[i] if i < len(all_stage_entropy_histories) else [[], [], [], []]
                            stage_minp_histories = all_stage_minp_histories[i] if i < len(all_stage_minp_histories) else [[], [], [], []]
                            
                            # 合并所有阶段的响应
                            combined_response = " ".join(stage_responses)
                            full_response = extract_answer_content(combined_response)
                            predicted_answer = extract_answer(full_response, dataset_type)
                            
                            # 合并熵值和min_p历史
                            combined_entropy_history = []
                            combined_minp_history = []
                            for stage_entropy, stage_minp in zip(stage_entropy_histories, stage_minp_histories):
                                combined_entropy_history.extend(stage_entropy)
                                combined_minp_history.extend(stage_minp)
                            
                            trial_results['responses'].append(full_response)
                            trial_results['entropy_histories'].append(combined_entropy_history)
                            trial_results['minp_histories'].append(combined_minp_history)
                            trial_results['stage_responses'].append(stage_responses)
                            trial_results['stage_entropy_histories'].append(stage_entropy_histories)
                            trial_results['stage_minp_histories'].append(stage_minp_histories)
                            trial_results['predicted_answers'].append(predicted_answer)
                            trial_results['ground_truth_answers'].append(item.get("answer", ""))
                            trial_results['generation_times'].append((time.time() - start_time) / (i + 1))
                        else:
                            # 处理缺失的结果
                            trial_results['responses'].append("")
                            trial_results['entropy_histories'].append([])
                            trial_results['minp_histories'].append([])
                            trial_results['stage_responses'].append(["", "", "", ""])
                            trial_results['stage_entropy_histories'].append([[], [], [], []])
                            trial_results['stage_minp_histories'].append([[], [], [], []])
                            trial_results['predicted_answers'].append("")
                            trial_results['ground_truth_answers'].append(item.get("answer", ""))
                            trial_results['generation_times'].append(0)
                else:
                    # 逐个4阶段生成
                    logger.info(f"使用逐个4阶段CoT生成")
                    for i, item in enumerate(dataset_items):
                        logger.info(f"问题 {i + 1}/{len(dataset_items)}: 进行4阶段生成...")
                        
                        question = item["question"]
                        stage_responses, stage_entropy_histories, stage_minp_histories = multi_stage_generate(
                            model, tokenizer, question, dataset_type, dynamic_min_p_processor, generation_config, config
                        )
                        
                        # 合并所有阶段的响应
                        combined_response = " ".join(stage_responses)
                        full_response = extract_answer_content(combined_response)
                        predicted_answer = extract_answer(full_response, dataset_type)
                        
                        # 合并熵值和min_p历史
                        combined_entropy_history = []
                        combined_minp_history = []
                        for stage_entropy, stage_minp in zip(stage_entropy_histories, stage_minp_histories):
                            combined_entropy_history.extend(stage_entropy)
                            combined_minp_history.extend(stage_minp)
                        
                        trial_results['responses'].append(full_response)
                        trial_results['entropy_histories'].append(combined_entropy_history)
                        trial_results['minp_histories'].append(combined_minp_history)
                        trial_results['stage_responses'].append(stage_responses)
                        trial_results['stage_entropy_histories'].append(stage_entropy_histories)
                        trial_results['stage_minp_histories'].append(stage_minp_histories)
                        trial_results['predicted_answers'].append(predicted_answer)
                        trial_results['ground_truth_answers'].append(item.get("answer", ""))
                        trial_results['generation_times'].append((time.time() - start_time) / (i + 1))
            
            else:
                # 简单CoT或无CoT生成
                if enable_batch:
                    logger.info(f"使用批量{cot_method}生成，批量大小: {batch_size}")
                    all_responses, all_entropy_histories, all_minp_histories = batch_generate(
                        model, tokenizer, questions, dynamic_min_p_processor, generation_config, batch_size
                    )
                    
                    # 处理批量结果
                    for i, item in enumerate(dataset_items):
                        if i < len(all_responses):
                            response = all_responses[i]
                            entropy_history = all_entropy_histories[i] if i < len(all_entropy_histories) else []
                            minp_history = all_minp_histories[i] if i < len(all_minp_histories) else []
                            
                            full_response = extract_answer_content(response)
                            predicted_answer = extract_answer(full_response, dataset_type)
                            
                            trial_results['responses'].append(full_response)
                            trial_results['entropy_histories'].append(entropy_history)
                            trial_results['minp_histories'].append(minp_history)
                            trial_results['predicted_answers'].append(predicted_answer)
                            trial_results['ground_truth_answers'].append(item.get("answer", ""))
                            trial_results['generation_times'].append((time.time() - start_time) / (i + 1))
                        else:
                            # 处理缺失的结果
                            trial_results['responses'].append("")
                            trial_results['entropy_histories'].append([])
                            trial_results['minp_histories'].append([])
                            trial_results['predicted_answers'].append("")
                            trial_results['ground_truth_answers'].append(item.get("answer", ""))
                            trial_results['generation_times'].append(0)
                
                else:
                    # 逐个简单生成
                    logger.info(f"使用逐个{cot_method}生成")
                    for i, item in enumerate(dataset_items):
                        logger.info(f"问题 {i + 1}/{len(dataset_items)}: 进行{cot_method}生成...")
                        
                        question = questions[i]
                        
                        # 重置处理器状态
                        dynamic_min_p_processor.reset_entropy_history()
                        
                        # 生成响应
                        inputs = tokenizer(question, return_tensors="pt").to(model.device)
                        
                        # 过滤掉模型不支持的参数
                        filtered_config = {k: v for k, v in generation_config.items() 
                                         if k not in ['batch_size', 'enable_batch_generation', 'cot_method']}
                        
                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                logits_processor=[dynamic_min_p_processor],
                                **filtered_config
                            )
                        
                        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                        full_response = extract_answer_content(response)
                        predicted_answer = extract_answer(full_response, dataset_type)
                        
                        # 获取熵值和min_p历史
                        entropy_history = dynamic_min_p_processor.get_entropy_history()
                        minp_history = dynamic_min_p_processor.get_minp_history()
                        
                        trial_results['responses'].append(full_response)
                        trial_results['entropy_histories'].append(entropy_history)
                        trial_results['minp_histories'].append(minp_history)
                        trial_results['predicted_answers'].append(predicted_answer)
                        trial_results['ground_truth_answers'].append(item.get("answer", ""))
                        trial_results['generation_times'].append((time.time() - start_time) / (i + 1))
            
            total_time = time.time() - start_time
            
            # 计算准确率和其他统计信息
            if trial_results['predicted_answers'] and trial_results['ground_truth_answers']:
                accuracy, correct_predictions = calculate_accuracy(
                    trial_results['predicted_answers'], 
                    trial_results['ground_truth_answers']
                )
                trial_results['accuracy'] = accuracy
                trial_results['correct_predictions'] = correct_predictions
                
                # 统计正确和错误的样本数量
                correct_count = sum(correct_predictions)
                total_count = len(correct_predictions)
                
                logger.info(f"\n=== 试验 {trial_idx + 1} 结果统计 ===")
                logger.info(f"总样本数: {total_count}")
                logger.info(f"正确数量: {correct_count}")
                logger.info(f"错误数量: {total_count - correct_count}")
                logger.info(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
                logger.info(f"平均生成时间: {sum(trial_results['generation_times'])/len(trial_results['generation_times']):.2f}秒/样本")
                logger.info(f"总生成时间: {total_time:.2f}秒")
                
                # 显示错误样本的详细信息
                error_samples = []
                for i, (is_correct, pred, truth) in enumerate(zip(correct_predictions, trial_results['predicted_answers'], trial_results['ground_truth_answers'])):
                    if not is_correct:
                        error_samples.append({
                            'index': i,
                            'question': dataset_items[i]['question'],
                            'predicted': pred,
                            'ground_truth': truth,
                            'response': trial_results['responses'][i] if i < len(trial_results['responses']) else ""
                        })
                
                if error_samples:
                    logger.info(f"\n--- 错误样本详情 (共{len(error_samples)}个) ---")
                    for error in error_samples[:5]:  # 只显示前5个错误样本
                        logger.info(f"\n❌ 样本 {error['index'] + 1}:")
                        logger.info(f"问题: {error['question']}")
                        logger.info(f"标准答案: {error['ground_truth']}")
                        logger.info(f"预测答案: {error['predicted']}")
                        logger.info(f"完整响应: {error['response']}")
                    
                    if len(error_samples) > 5:
                        logger.info(f"... 还有 {len(error_samples) - 5} 个错误样本未显示")
            else:
                trial_results['accuracy'] = 0.0
                trial_results['correct_predictions'] = []
                logger.warning(f"试验 {trial_idx + 1} 缺少预测答案或标准答案，无法计算准确率")
        
        # 将当前试验结果添加到所有试验结果中
        all_trials_results.append(trial_results)
    
    # 计算多次试验的统计结果
    if num_trials > 1:
        accuracies = [trial.get('accuracy', 0.0) for trial in all_trials_results]
        average_accuracy = sum(accuracies) / len(accuracies)
        accuracy_std = (sum((acc - average_accuracy) ** 2 for acc in accuracies) / len(accuracies)) ** 0.5
        
        generation_times_per_trial = []
        for trial in all_trials_results:
            if trial['generation_times']:
                avg_time = sum(trial['generation_times']) / len(trial['generation_times'])
                generation_times_per_trial.append(avg_time)
        
        logger.info(f"\n=== 多次试验汇总统计 ===")
        logger.info(f"试验次数: {num_trials}")
        logger.info(f"平均准确率: {average_accuracy:.4f} ± {accuracy_std:.4f}")
        logger.info(f"各次试验准确率: {[f'{acc:.4f}' for acc in accuracies]}")
        if generation_times_per_trial:
            logger.info(f"平均生成时间: {sum(generation_times_per_trial)/len(generation_times_per_trial):.2f}秒/样本")
        
        return {
            'all_trials': all_trials_results,
            'average_accuracy': average_accuracy,
            'accuracy_std': accuracy_std,
            'individual_accuracies': accuracies,
            'average_generation_times': generation_times_per_trial
        }
    else:
        return {
            'all_trials': all_trials_results,
            'average_accuracy': all_trials_results[0].get('accuracy', 0.0),
            'accuracy_std': 0.0,
            'individual_accuracies': [all_trials_results[0].get('accuracy', 0.0)],
            'average_generation_times': [sum(all_trials_results[0]['generation_times'])/len(all_trials_results[0]['generation_times'])] if all_trials_results[0]['generation_times'] else [0.0]
        }

def truncate_stage_response(response: str, current_stage: int) -> str:
    """
    截断阶段响应，避免生成下一阶段的内容
    
    Args:
        response: 原始响应
        current_stage: 当前阶段号 (1-4)
        
    Returns:
        截断后的响应
    """
    if not response:
        return response
    
    logger = logging.getLogger("dynamics_minp")
    original_response = response
    
    # 定义所有可能的阶段标识符（不仅仅是下一阶段）
    all_stage_patterns = []
    for stage_num in range(1, 5):  # Step 1-4
        if stage_num != current_stage:  # 排除当前阶段
            all_stage_patterns.extend([
                f"\nStep {stage_num}:",
                f"\nstep {stage_num}:",
                f"\nSTEP {stage_num}:",
                f"Step {stage_num}:",
                f"step {stage_num}:",
                f"STEP {stage_num}:",
                f"阶段{stage_num}:",
                f"第{stage_num}步:",
                f"## Step {stage_num}",
                f"**Step {stage_num}**"
            ])
    
    # 检查所有可能的截断位置，选择最早的
    earliest_pos = len(response)
    truncation_reason = None
    
    # 检查阶段标识符
    for pattern in all_stage_patterns:
        if pattern in response:
            pattern_pos = response.find(pattern)
            if pattern_pos < earliest_pos:
                earliest_pos = pattern_pos
                truncation_reason = f"阶段标识符 '{pattern}'"
    
    # 检查双换行符 - 但要确保不是在很开始的位置（避免过度截断）
    if "\n\n" in response:
        double_newline_pos = response.find("\n\n")
        # 只有在双换行符不在开头附近时才截断（至少要有30个字符的内容）
        if double_newline_pos >= 30 and double_newline_pos < earliest_pos:
            earliest_pos = double_newline_pos
            truncation_reason = "双换行符"
    
    # 检查其他可能的分段标识符
    other_separators = [
        "\n\nStep",  # 双换行后直接跟Step
        "\n\nAnswer:",  # 双换行后跟Answer:
        "\n\nThe final answer",  # 双换行后跟最终答案
        "\n\nFinal answer",
        "\n\n$\\boxed",  # 双换行后跟boxed答案
    ]
    
    for separator in other_separators:
        if separator in response:
            sep_pos = response.find(separator)
            if sep_pos < earliest_pos:
                earliest_pos = sep_pos
                truncation_reason = f"分段标识符 '{separator}'"
    
    # 执行截断
    if earliest_pos < len(response):
        if logger.handlers:
            logger.debug(f"  截断检测到{truncation_reason}在位置 {earliest_pos}")
        response = response[:earliest_pos]
    
    # 额外的安全措施：限制长度，防止生成过长内容
    lines = response.split('\n')
    if len(lines) > 10:  # 限制每个阶段最多10行
        if logger.handlers:
            logger.debug(f"  截断因行数过多 ({len(lines)}行 > 10行)")
        response = '\n'.join(lines[:10])
    
    if logger.handlers and response != original_response:
        logger.debug(f"  截断前长度: {len(original_response)}, 截断后长度: {len(response)}")
    
    return response.strip()


def extract_generated_content(stage_response: str, stage_prompt: str) -> str:
    """
    从阶段响应中提取生成的内容（去掉提示词前缀）
    
    Args:
        stage_response: 完整的阶段响应（可能包含提示词前缀）
        stage_prompt: 阶段提示词
        
    Returns:
        提取的生成内容
    """
    if not stage_response or not stage_response.strip():
        return ""
    
    # 尝试多种方式匹配和提取
    stage_response = stage_response.strip()
    
    # 方式1：直接匹配提示词前缀
    if stage_response.startswith(stage_prompt):
        return stage_response[len(stage_prompt):].strip()
    
    # 方式2：查找提示词在响应中的位置
    if stage_prompt in stage_response:
        idx = stage_response.find(stage_prompt)
        return stage_response[idx + len(stage_prompt):].strip()
    
    # 方式3：如果都不匹配，返回原始内容
    return stage_response


def extract_answer_content(full_text: str) -> str:
    """
    从完整文本中提取 "Answer:" 后面的内容
    
    Args:
        full_text: 完整的生成文本
        
    Returns:
        str: Answer: 后面的内容，如果没有找到则返回原文本
    """
    if not full_text:
        return full_text
    
    # 查找 "Answer:" 的位置（不区分大小写）
    answer_patterns = ["Answer:", "answer:", "ANSWER:", "Answer：", "answer："]
    
    for pattern in answer_patterns:
        if pattern in full_text:
            # 找到第一个匹配的位置
            idx = full_text.find(pattern)
            # 返回该位置之后的内容，去除前后空白
            return full_text[idx + len(pattern):].strip()
    
    # 如果没有找到 "Answer:" 标记，返回原文本
    return full_text


def majority_vote(answers: List[str], dataset_type: str, confidence_threshold: float = 0.0) -> tuple[str, dict]:
    """
    对多个答案进行多数投票，支持置信度阈值
    
    Args:
        answers: 答案列表
        dataset_type: 数据集类型
        confidence_threshold: 置信度阈值，只有超过此阈值的答案才会被采纳
        
    Returns:
        (最终答案, 投票详情字典)
    """
    if not answers:
        return "", {"vote_counts": {}, "total_votes": 0, "winner": "", "confidence": 0.0, "meets_threshold": False, "threshold": confidence_threshold}
    
    # 提取所有答案的最终结果
    extracted_answers = []
    for answer in answers:
        extracted = extract_answer(answer, dataset_type)
        extracted_answers.append(str(extracted).strip().lower())
    
    # 统计投票
    vote_counts = {}
    for answer in extracted_answers:
        vote_counts[answer] = vote_counts.get(answer, 0) + 1
    
    # 找出得票最多的答案
    if not vote_counts:
        return "", {"vote_counts": {}, "total_votes": 0, "winner": "", "confidence": 0.0, "meets_threshold": False, "threshold": confidence_threshold}
    
    winner = max(vote_counts.items(), key=lambda x: x[1])
    winner_answer, winner_votes = winner
    total_votes = len(extracted_answers)
    confidence = winner_votes / total_votes
    
    # 检查是否满足置信度阈值
    meets_threshold = confidence >= confidence_threshold
    
    voting_details = {
        "vote_counts": vote_counts,
        "total_votes": total_votes,
        "winner": winner_answer,
        "winner_votes": winner_votes,
        "confidence": confidence,
        "meets_threshold": meets_threshold,
        "threshold": confidence_threshold
    }
    
    return winner_answer, voting_details


def generate_greedy_fallback(model, tokenizer, input_text: str, dynamic_min_p_processor, generation_config: Dict, dataset_type: str) -> str:
    """
    当投票置信度不足时，使用贪婪解码生成回退答案
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        input_text: 输入文本
        dynamic_min_p_processor: 动态min_p处理器
        generation_config: 生成配置
        dataset_type: 数据集类型
        
    Returns:
        贪婪解码生成的答案
    """
    logger = logging.getLogger("dynamics_minp")
    logger.info(f"  🔄 投票置信度不足，启用贪婪解码回退...")
    
    # 重置处理器状态
    dynamic_min_p_processor.reset_entropy_history()
    
    # 创建贪婪解码配置
    greedy_config = {
        k: v for k, v in generation_config.items() 
        if k in ['max_new_tokens', 'top_p', 'top_k', 'num_beams', 'repetition_penalty']
    }
    # 贪婪解码设置
    greedy_config.update({
        'do_sample': False,  # 关闭采样，使用贪婪解码
        'temperature': 1.0,  # 贪婪解码时温度无效，但设置为标准值
        'num_beams': 1       # 使用单束搜索
    })
    
    # 获取模型设备
    device = next(model.parameters()).device
    
    try:
        # 分词
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        # 贪婪生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                logits_processor=[dynamic_min_p_processor],
                **greedy_config
            )
        
        # 解码响应
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 提取生成的部分
        if response.startswith(input_text):
            generated_response = response[len(input_text):].strip()
        else:
            generated_response = response.strip()
        
        # 提取答案
        final_answer = extract_answer(generated_response, dataset_type)
        
        logger.info(f"  ✓ 贪婪解码完成，生成答案: {final_answer}")
        return str(final_answer).strip().lower()
        
    except Exception as e:
        logger.error(f"贪婪解码回退失败: {e}")
        return ""


def generate_multiple_responses(model, tokenizer, input_text: str, dynamic_min_p_processor, generation_config: Dict, num_votes: int, voting_temperature: float = 0.8) -> List[str]:
    """
    为单个输入生成多个响应用于投票（串行版本，保持向后兼容）
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        input_text: 输入文本
        dynamic_min_p_processor: 动态min_p处理器
        generation_config: 生成配置
        num_votes: 投票次数
        voting_temperature: 投票时使用的温度
        
    Returns:
        响应列表
    """
    responses = []
    
    # 使用投票温度
    voting_config = generation_config.copy()
    voting_config['temperature'] = voting_temperature
    
    # 过滤掉模型不支持的参数
    filtered_config = {k: v for k, v in voting_config.items() 
                      if k not in ['batch_size', 'enable_batch_generation', 'cot_method']}
    
    for i in range(num_votes):
        dynamic_min_p_processor.reset_entropy_history()
        
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                logits_processor=[dynamic_min_p_processor],
                **filtered_config
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        responses.append(response)
    
    return responses


def generate_multiple_responses_parallel(model, tokenizer, input_text: str, dynamic_min_p_processor, generation_config: Dict, num_votes: int, voting_temperature: float = 0.8) -> List[str]:
    """
    为单个输入生成多个响应用于投票（并行版本）
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        input_text: 输入文本
        dynamic_min_p_processor: 动态min_p处理器
        generation_config: 生成配置
        num_votes: 投票次数
        voting_temperature: 投票时使用的温度
        
    Returns:
        响应列表
    """
    logger = logging.getLogger("dynamics_minp")
    if logger.handlers:
        logger.info(f"  正在并行生成 {num_votes} 个候选...")
    
    # 使用投票温度
    voting_config = generation_config.copy()
    voting_config['temperature'] = voting_temperature
    
    # 创建清理后的生成配置（只保留模型支持的参数）
    clean_gen_config = {
        k: v for k, v in voting_config.items() 
        if k in ['max_new_tokens', 'temperature', 'do_sample', 'top_p', 'top_k', 'num_beams', 'repetition_penalty']
    }
    
    # 获取模型设备
    device = next(model.parameters()).device
    
    # 构建批量输入（所有候选使用相同的输入）
    batch_inputs = [input_text for _ in range(num_votes)]
    
    # 批量分词
    tokenized_inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True)
    
    # 将输入张量移动到模型设备
    tokenized_inputs = {k: v.to(device) for k, v in tokenized_inputs.items()}
    
    # 重置处理器状态
    dynamic_min_p_processor.reset_entropy_history()
    
    # 并行生成所有候选响应
    with torch.no_grad():
        outputs = model.generate(
            **tokenized_inputs,
            logits_processor=[dynamic_min_p_processor],
            **clean_gen_config
        )
    
    # 解码所有响应并提取生成的部分
    responses = []
    for vote_idx in range(num_votes):
        response = tokenizer.decode(outputs[vote_idx], skip_special_tokens=True)
        generated_response = response[len(batch_inputs[vote_idx]):].strip()
        responses.append(generated_response)
    
    return responses


def generate_batch_voting_responses(model, tokenizer, questions: List[str], dataset_type: str, dynamic_min_p_processor, generation_config: Dict, config: Dict, num_votes: int, voting_temperature: float = 0.8, cot_method: str = "simple", batch_size: int = 4) -> Tuple[List[List[str]], List[List]]:
    """
    批量并行投票生成：同时处理多个问题的多次投票
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        questions: 问题列表
        dataset_type: 数据集类型
        dynamic_min_p_processor: 动态min_p处理器
        generation_config: 生成配置
        config: 完整配置
        num_votes: 每个问题的投票次数
        voting_temperature: 投票温度
        cot_method: CoT方法
        batch_size: 批量大小
        
    Returns:
        Tuple[List[List[str]], List[List]]: (每个问题的候选响应列表, 额外信息列表)
    """
    logger = logging.getLogger("dynamics_minp")
    all_question_responses = []
    all_question_extra_info = []
    
    # 分批处理问题
    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i:i + batch_size]
        current_batch_size = len(batch_questions)
        
        logger.info(f"批量并行投票 - 批次 {i//batch_size + 1}/{(len(questions) + batch_size - 1)//batch_size}")
        logger.info(f"  处理 {current_batch_size} 个问题，每个问题 {num_votes} 次投票")
        logger.info(f"  总计 {current_batch_size * num_votes} 个并行生成任务")
        
        if cot_method == "4_stage_cot":
            # 4阶段CoT的批量投票 - 暂时使用原有逻辑
            batch_responses = []
            batch_extra_info = []
            for question in batch_questions:
                responses, extra_info = generate_multiple_responses_universal(
                    model, tokenizer, question, dataset_type, dynamic_min_p_processor,
                    generation_config, config, num_votes, voting_temperature, cot_method, True
                )
                batch_responses.append(responses)
                batch_extra_info.append(extra_info)
            
            all_question_responses.extend(batch_responses)
            all_question_extra_info.extend(batch_extra_info)
        
        else:
            # 简单CoT和无CoT的超级批量并行
            # 构建所有候选的输入：batch_size * num_votes 个输入
            mega_batch_inputs = []
            question_indices = []  # 记录每个输入对应的问题索引
            
            for q_idx, question in enumerate(batch_questions):
                enhanced_question = enhance_prompt_with_cot(config, question, dataset_type, cot_method)
                for vote_idx in range(num_votes):
                    mega_batch_inputs.append(enhanced_question)
                    question_indices.append(q_idx)
            
            logger.info(f"  构建了 {len(mega_batch_inputs)} 个并行输入")
            
            # 超级批量生成
            mega_responses = batch_generate_mega(
                model, tokenizer, mega_batch_inputs, dynamic_min_p_processor, 
                generation_config, voting_temperature
            )
            
            # 将响应按问题重新组织
            batch_responses = [[] for _ in range(current_batch_size)]
            for resp_idx, response in enumerate(mega_responses):
                q_idx = question_indices[resp_idx]
                batch_responses[q_idx].append(response)
            
            all_question_responses.extend(batch_responses)
            all_question_extra_info.extend([[] for _ in range(current_batch_size)])
    
    return all_question_responses, all_question_extra_info


def batch_generate_mega(model, tokenizer, inputs: List[str], dynamic_min_p_processor, generation_config: Dict, temperature: float) -> List[str]:
    """
    超级批量生成：一次性处理大量输入
    """
    logger = logging.getLogger("dynamics_minp")
    
    # 重置处理器状态
    dynamic_min_p_processor.reset_entropy_history()
    
    # 创建清理后的生成配置
    clean_gen_config = {
        k: v for k, v in generation_config.items() 
        if k in ['max_new_tokens', 'do_sample', 'top_p', 'top_k', 'num_beams', 'repetition_penalty']
    }
    clean_gen_config['temperature'] = temperature
    
    # 获取模型设备
    device = next(model.parameters()).device
    
    try:
        # 批量分词
        tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
        tokenized_inputs = {k: v.to(device) for k, v in tokenized_inputs.items()}
        
        logger.info(f"  开始超级批量生成 {len(inputs)} 个候选...")
        
        # 并行生成所有响应
        with torch.no_grad():
            outputs = model.generate(
                **tokenized_inputs,
                logits_processor=[dynamic_min_p_processor],
                **clean_gen_config
            )
        
        # 解码所有响应
        responses = []
        for i, output in enumerate(outputs):
            response = tokenizer.decode(output, skip_special_tokens=True)
            # 提取生成的部分
            original_input = inputs[i]
            if response.startswith(original_input):
                generated_response = response[len(original_input):].strip()
            else:
                generated_response = response.strip()
            responses.append(generated_response)
        
        logger.info(f"  ✓ 超级批量生成完成")
        return responses
        
    except Exception as e:
        logger.error(f"超级批量生成失败: {e}")
        # 回退到逐个生成
        logger.info("回退到逐个生成模式...")
        responses = []
        for input_text in inputs:
            try:
                dynamic_min_p_processor.reset_entropy_history()
                tokenized_input = tokenizer(input_text, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    output = model.generate(
                        **tokenized_input,
                        logits_processor=[dynamic_min_p_processor],
                        **clean_gen_config
                    )
                
                response = tokenizer.decode(output[0], skip_special_tokens=True)
                generated_response = response[len(input_text):].strip()
                responses.append(generated_response)
                
            except Exception as single_e:
                logger.error(f"单个生成也失败: {single_e}")
                responses.append("")
        
        return responses


def generate_multiple_responses_universal(model, tokenizer, question: str, dataset_type: str, dynamic_min_p_processor, generation_config: Dict, config: Dict, num_votes: int, voting_temperature: float = 0.8, cot_method: str = "simple", use_parallel: bool = True, question_idx: int = 0, total_questions: int = 0) -> Tuple[List[str], List]:
    """
    通用的多响应生成函数，支持所有CoT模式和并行/串行选择
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        question: 问题
        dataset_type: 数据集类型
        dynamic_min_p_processor: 动态min_p处理器
        generation_config: 生成配置
        config: 完整配置
        num_votes: 投票次数
        voting_temperature: 投票时使用的温度
        cot_method: CoT方法 ("4_stage_cot", "simple", "none")
        use_parallel: 是否使用并行生成
        
    Returns:
        Tuple[List[str], List]: (完整响应列表, 额外信息列表)
    """
    logger = logging.getLogger("dynamics_minp")
    
    if cot_method == "4_stage_cot":
        # 4阶段CoT生成
        if use_parallel:
            logger.info(f"问题 {question_idx + 1}/{total_questions}: 为问题进行 {num_votes} 次4阶段并行投票生成...")
            combined_responses, all_stage_responses = generate_multiple_4stage_responses(
                model, tokenizer, question, dataset_type, dynamic_min_p_processor,
                generation_config, config, num_votes, voting_temperature
            )
            return combined_responses, all_stage_responses
        else:
            logger.info(f"问题 {question_idx + 1}/{total_questions}: 为问题进行 {num_votes} 次4阶段串行投票生成...")
            responses = []
            all_stage_responses = []
            for i in range(num_votes):
                stage_responses, _, _ = multi_stage_generate(
                    model, tokenizer, question, dataset_type, dynamic_min_p_processor, generation_config, config
                )
                combined_response = " ".join(stage_responses)
                responses.append(combined_response)
                all_stage_responses.append(stage_responses)
            return responses, all_stage_responses
    
    else:
        # 简单CoT或无CoT生成
        enhanced_question = enhance_prompt_with_cot(config, question, dataset_type, cot_method)
        
        if use_parallel:
            logger.info(f"问题 {question_idx + 1}/{total_questions}: 为问题进行 {num_votes} 次{cot_method}并行投票生成...")
            responses = generate_multiple_responses_parallel(
                model, tokenizer, enhanced_question, dynamic_min_p_processor,
                generation_config, num_votes, voting_temperature
            )
        else:
            logger.info(f"问题 {question_idx + 1}/{total_questions}: 为问题进行 {num_votes} 次{cot_method}串行投票生成...")
            responses = generate_multiple_responses(
                model, tokenizer, enhanced_question, dynamic_min_p_processor,
                generation_config, num_votes, voting_temperature
            )
        
        return responses, []  # 非4阶段CoT没有额外的阶段信息


def generate_multiple_4stage_responses(model, tokenizer, question: str, dataset_type: str, dynamic_min_p_processor, generation_config: Dict, config: Dict, num_votes: int, voting_temperature: float = 0.8) -> Tuple[List[str], List[List[str]]]:
    """
    为4阶段CoT生成多个响应用于投票（并行批量生成版本）
    
    Args:
        model: 语言模型
        tokenizer: 分词器  
        question: 问题
        dataset_type: 数据集类型
        dynamic_min_p_processor: 动态min_p处理器
        generation_config: 生成配置
        config: 完整配置
        num_votes: 投票次数
        voting_temperature: 投票时使用的温度
        
    Returns:
        Tuple[List[str], List[List[str]]]: (完整响应列表, 每个路径的阶段响应列表)
    """
    # 备份原始温度并使用投票温度
    voting_gen_config = generation_config.copy()
    voting_gen_config['temperature'] = voting_temperature
    
    # 获取4阶段提示词模板
    stage_prompts = get_stage_prompts(config, question, dataset_type)
    
    # 创建清理后的生成配置（只保留模型支持的参数）
    clean_gen_config = {
        k: v for k, v in voting_gen_config.items() 
        if k in ['max_new_tokens', 'temperature', 'do_sample', 'top_p', 'top_k', 'num_beams', 'repetition_penalty']
    }
    
    # 为4阶段CoT调整每个阶段的token限制
    if 'max_new_tokens' in clean_gen_config:
        clean_gen_config['max_new_tokens'] = clean_gen_config['max_new_tokens'] // 4
    
    # 获取模型设备
    device = next(model.parameters()).device
    
    # 初始化所有路径的状态
    all_stage_responses = [[] for _ in range(num_votes)]  # 每个路径的阶段响应
    conversation_histories = ["" for _ in range(num_votes)]  # 每个路径的对话历史
    
    # 逐阶段并行生成
    for stage_idx, stage_prompt in enumerate(stage_prompts):
        logger = logging.getLogger("dynamics_minp")
        if logger.handlers:
            logger.info(f"  阶段 {stage_idx + 1}/4: 正在并行生成 {num_votes} 个候选...")
        
        # 构建所有路径的输入
        batch_inputs = []
        for vote_idx in range(num_votes):
            full_input = conversation_histories[vote_idx] + stage_prompt
            batch_inputs.append(full_input)
        
        # 批量分词
        tokenized_inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True)
        
        # 将输入张量移动到模型设备
        tokenized_inputs = {k: v.to(device) for k, v in tokenized_inputs.items()}
        
        # 重置处理器状态（为当前阶段的批量生成做准备）
        dynamic_min_p_processor.reset_entropy_history()
        
        # 并行生成当前阶段的所有候选响应
        with torch.no_grad():
            outputs = model.generate(
                **tokenized_inputs,
                logits_processor=[dynamic_min_p_processor],
                **clean_gen_config
            )
        
        # 解码所有响应并提取生成的部分
        for vote_idx in range(num_votes):
            response = tokenizer.decode(outputs[vote_idx], skip_special_tokens=True)
            stage_response = response[len(batch_inputs[vote_idx]):].strip()
            
            # 截断阶段响应，避免生成下一阶段内容
            truncated_response = truncate_stage_response(stage_response, stage_idx + 1)
            
            # 保存当前阶段的响应
            all_stage_responses[vote_idx].append(truncated_response)
            
            # 更新对话历史
            conversation_histories[vote_idx] = batch_inputs[vote_idx] + truncated_response + "\n"
    
    # 合并所有路径的完整响应（包含4阶段引导提示词）
    full_responses = []
    for vote_idx in range(num_votes):
        combined_parts = []
        for stage_idx, (stage_prompt, stage_resp) in enumerate(zip(stage_prompts, all_stage_responses[vote_idx])):
            if stage_idx == 0:
                # 第一阶段：从完整提示中提取阶段提示部分
                stage_prompt_only = stage_prompt.split('\n\n')[-1] if '\n\n' in stage_prompt else stage_prompt
            else:
                # 后续阶段：直接使用提示词
                stage_prompt_only = stage_prompt
                
            combined_parts.append(f"{stage_prompt_only} {stage_resp}")
        
        combined_response = " ".join(combined_parts)
        full_responses.append(combined_response)
    
    return full_responses, all_stage_responses


def calculate_accuracy(predicted_answers: List[str], ground_truth_answers: List[str]) -> Tuple[float, List[bool]]:
    """
    计算预测答案的准确率
    
    Args:
        predicted_answers: 预测答案列表
        ground_truth_answers: 标准答案列表
        
    Returns:
        Tuple[float, List[bool]]: (准确率, 每个样本的正确性列表)
    """
    if len(predicted_answers) != len(ground_truth_answers):
        raise ValueError("预测答案数量与标准答案数量不匹配")
    
    correct_predictions = []
    for pred, truth in zip(predicted_answers, ground_truth_answers):
        # 标准化答案进行比较
        pred_clean = str(pred).strip().lower()
        truth_clean = str(truth).strip().lower()
        
        # 对于数值答案，尝试数值比较
        try:
            pred_num = float(pred_clean)
            truth_num = float(truth_clean)
            is_correct = abs(pred_num - truth_num) < 1e-6  # 考虑浮点精度
        except (ValueError, TypeError):
            # 字符串比较
            is_correct = pred_clean == truth_clean
        
        correct_predictions.append(is_correct)
    
    accuracy = sum(correct_predictions) / len(correct_predictions)
    return accuracy, correct_predictions


def save_results(results: List[Dict], output_path: str):
    """
    保存实验结果到JSON文件
    
    Args:
        results: 结果列表
        output_path: 输出文件路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到: {output_path}")


def multi_stage_generate(model, tokenizer, question: str, dataset_type: str, dynamic_min_p_processor, generation_config: Dict, config: Dict) -> Tuple[List[str], List[List[float]], List[List[float]]]:
    """
    多阶段生成：分4次生成，每次有不同的引导
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        question: 问题
        dataset_type: 数据集类型
        dynamic_min_p_processor: 动态min_p处理器
        generation_config: 生成配置
    
    Returns:
        Tuple[List[str], List[List[float]], List[List[float]]]: (每个阶段的响应列表, 每个阶段的熵值历史列表, 每个阶段的min_p历史列表)
    """
    stage_prompts = get_stage_prompts(config, question, dataset_type)
    stage_responses = []
    stage_entropy_histories = []
    stage_minp_histories = []
    
    # 构建对话历史
    conversation_history = ""
    
    for stage_idx, stage_prompt in enumerate(stage_prompts):
        logger = logging.getLogger("dynamics_minp")
        if logger.handlers:
            logger.info(f"  阶段 {stage_idx + 1}/4: 正在生成...")
        else:
            print(f"  阶段 {stage_idx + 1}/4: 正在生成...")
        
        # 重置熵和min_p记录
        dynamic_min_p_processor.reset_entropy_history()
        
        # 构建当前阶段的完整输入
        if stage_idx == 0:
            # 第一阶段直接使用阶段提示
            current_input = stage_prompt
        else:
            # 后续阶段直接继续之前的内容，不添加额外换行
            current_input = conversation_history + stage_prompt
        
        # 生成当前阶段的响应
        inputs = tokenizer(current_input, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=min(128, generation_config.get('max_new_tokens', 512) // 4),  # 每个阶段最多128个token
                logits_processor=LogitsProcessorList([dynamic_min_p_processor]),
                do_sample=generation_config.get('do_sample', True),
                temperature=generation_config.get('temperature', 0.9),
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 解码生成的文本
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        # 移除输入部分，只保留生成的部分
        stage_response = generated_text[len(current_input):].strip()
        # 截断阶段响应，避免生成下一阶段内容
        truncated_response = truncate_stage_response(stage_response, stage_idx + 1)
        
        # 存储时添加阶段提示词前缀
        if stage_idx == 0:
            # 阶段1：从完整提示中提取阶段提示部分
            stage_prompt_only = stage_prompts[stage_idx].split('\n\n')[-1]
        else:
            # 阶段2-4：直接使用完整的阶段提示
            stage_prompt_only = stage_prompts[stage_idx]
        
        stage_with_prompt = f"{stage_prompt_only} {truncated_response}"
        stage_responses.append(stage_with_prompt)
        stage_entropy_histories.append(dynamic_min_p_processor.get_entropy_history())
        stage_minp_histories.append(dynamic_min_p_processor.get_minp_history())
        
        # 更新对话历史 - 只包含提示词和响应，不重复问题
        if stage_idx == 0:
            conversation_history = f"{stage_prompt}\n{truncated_response}"
        else:
            conversation_history += f"\n{stage_prompt_only}\n{truncated_response}"
        
        # 只显示阶段进度，详细内容在错误分析时显示
        if logger.handlers:
            logger.info(f"  阶段 {stage_idx + 1}/4: 完成")
            # 调试级别的详细信息
            logger.debug(f"  阶段 {stage_idx + 1} 原始响应: {stage_response}")
            logger.debug(f"  阶段 {stage_idx + 1} 截断后响应: {truncated_response}")
            logger.debug(f"  阶段 {stage_idx + 1} 完整响应(含提示词): {stage_with_prompt}")
        else:
            print(f"  阶段 {stage_idx + 1}/4: 完成")
    
    return stage_responses, stage_entropy_histories, stage_minp_histories

def batch_multi_stage_generate(model, tokenizer, questions: List[str], dataset_type: str, dynamic_min_p_processor, generation_config: Dict, config: Dict, batch_size: int = 4, use_one_shot: bool = False, one_shot_prompts: str = "") -> Tuple[List[List[str]], List[List[List[float]]], List[List[List[float]]]]:
    """
    批量多阶段生成：对多个问题同时进行4阶段CoT生成
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        questions: 问题列表
        dataset_type: 数据集类型
        dynamic_min_p_processor: 动态min_p处理器
        generation_config: 生成配置
        config: 完整配置
        batch_size: 批量大小
        use_one_shot: 是否使用one-shot示例
        one_shot_prompts: one-shot提示词
    
    Returns:
        Tuple: (每个问题的阶段响应列表, 每个问题的阶段熵值历史列表, 每个问题的阶段min_p历史列表)
    """
    logger = logging.getLogger("dynamics_minp")
    logger.info(f"开始批量4阶段生成，共 {len(questions)} 个问题，批量大小: {batch_size}")
    
    all_stage_responses = []
    all_stage_entropy_histories = []
    all_stage_minp_histories = []
    
    # 获取4阶段提示词
    stage_prompts_template = get_stage_prompts(config, "", dataset_type)
    
    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i:i + batch_size]
        current_batch_size = len(batch_questions)
        logger.info(f"处理批次 {i//batch_size + 1}/{(len(questions) + batch_size - 1)//batch_size}，大小: {current_batch_size}")
        
        # 为当前批次初始化结果存储
        batch_stage_responses = [[] for _ in range(current_batch_size)]
        batch_stage_entropy_histories = [[] for _ in range(current_batch_size)]
        batch_stage_minp_histories = [[] for _ in range(current_batch_size)]
        
        # 构建当前批次的对话历史
        batch_conversation_history = ["" for _ in range(current_batch_size)]
        
        # 执行4个阶段
        for stage_idx in range(4):
            logger.info(f"  执行阶段 {stage_idx + 1}/4")
            
            # 构建当前阶段的提示词
            stage_prompts = []
            for q_idx, question in enumerate(batch_questions):
                if stage_idx == 0:
                    # 第一阶段：包含原始问题和one-shot示例
                    if use_one_shot and one_shot_prompts:
                        stage_prompt = f"{one_shot_prompts}\n\nNow solve this problem:\n\nQuestion: {question}\n\nAnswer: {stage_prompts_template[stage_idx]}"
                    else:
                        stage_prompt = f"{question}\n\n{stage_prompts_template[stage_idx]}"
                else:
                    # 后续阶段：基于之前的对话历史，但不重复"Answer:"标签
                    # 直接继续之前的内容
                    stage_prompt = f"{batch_conversation_history[q_idx]}{stage_prompts_template[stage_idx]}"
                
                stage_prompts.append(stage_prompt)
            
            # 重置处理器状态
            dynamic_min_p_processor.reset_entropy_history()
            
            try:
                # 批量生成当前阶段
                inputs = tokenizer(stage_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                # 过滤掉模型不支持的参数
                filtered_config = {k: v for k, v in generation_config.items() 
                                 if k not in ['batch_size', 'enable_batch_generation', 'cot_method']}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        logits_processor=[dynamic_min_p_processor],
                        **filtered_config
                    )
                
                # 解码响应
                stage_responses = []
                for j, output in enumerate(outputs):
                    input_length = inputs['input_ids'][j].shape[0]
                    response = tokenizer.decode(output[input_length:], skip_special_tokens=True)
                    # 截断阶段响应，避免生成下一阶段内容
                    truncated_response = truncate_stage_response(response, stage_idx + 1)
                    stage_responses.append(truncated_response)
                
                # 获取熵值和min_p历史
                batch_entropy_histories = dynamic_min_p_processor.get_batch_entropy_histories()
                batch_minp_histories = dynamic_min_p_processor.get_batch_minp_histories()
                
                # 存储结果并更新对话历史
                for j in range(current_batch_size):
                    # 存储时添加阶段提示词前缀
                    stage_with_prompt = f"{stage_prompts_template[stage_idx]} {stage_responses[j]}"
                    batch_stage_responses[j].append(stage_with_prompt)
                    batch_stage_entropy_histories[j].append(batch_entropy_histories[j] if j < len(batch_entropy_histories) else [])
                    batch_stage_minp_histories[j].append(batch_minp_histories[j] if j < len(batch_minp_histories) else [])
                    
                    # 更新对话历史
                    if stage_idx == 0:
                        if use_one_shot and one_shot_prompts:
                            batch_conversation_history[j] = f"{one_shot_prompts}\n\nNow solve this problem:\n\nQuestion: {batch_questions[j]}\n\nAnswer: {stage_with_prompt}"
                        else:
                            batch_conversation_history[j] = f"{batch_questions[j]}\n\n{stage_with_prompt}"
                    else:
                        batch_conversation_history[j] += f"\n\n{stage_with_prompt}"
                
                logger.info(f"    阶段 {stage_idx + 1} 完成")
                
            except Exception as e:
                logger.error(f"    阶段 {stage_idx + 1} 生成失败: {e}")
                # 如果批量生成失败，尝试单个生成
                for j in range(current_batch_size):
                    try:
                        if stage_idx == 0:
                            if use_one_shot and one_shot_prompts:
                                single_prompt = f"{one_shot_prompts}\n\nNow solve this problem:\n\nQuestion: {batch_questions[j]}\n\nAnswer: {stage_prompts_template[stage_idx]}"
                            else:
                                single_prompt = f"{batch_questions[j]}\n\n{stage_prompts_template[stage_idx]}"
                        else:
                            single_prompt = f"{batch_conversation_history[j]}\n\n{stage_prompts_template[stage_idx]}"
                        
                        # 单个生成
                        dynamic_min_p_processor.reset_entropy_history()
                        inputs = tokenizer(single_prompt, return_tensors="pt")
                        inputs = {k: v.to(model.device) for k, v in inputs.items()}
                        
                        # 过滤掉模型不支持的参数
                        filtered_config = {k: v for k, v in generation_config.items() 
                                         if k not in ['batch_size', 'enable_batch_generation', 'cot_method']}
                        
                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                logits_processor=[dynamic_min_p_processor],
                                **filtered_config
                            )
                        
                        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                        # 截断阶段响应，避免生成下一阶段内容
                        truncated_response = truncate_stage_response(response, stage_idx + 1)
                        entropy_history = dynamic_min_p_processor.get_entropy_history()
                        minp_history = dynamic_min_p_processor.get_minp_history()
                        
                        # 存储时添加阶段提示词前缀
                        stage_with_prompt = f"{stage_prompts_template[stage_idx]} {truncated_response}"
                        batch_stage_responses[j].append(stage_with_prompt)
                        batch_stage_entropy_histories[j].append(entropy_history)
                        batch_stage_minp_histories[j].append(minp_history)
                        
                        # 更新对话历史
                        if stage_idx == 0:
                            if use_one_shot and one_shot_prompts:
                                batch_conversation_history[j] = f"{one_shot_prompts}\n\nNow solve this problem:\n\nQuestion: {batch_questions[j]}\n\nAnswer: {stage_with_prompt}"
                            else:
                                batch_conversation_history[j] = f"{batch_questions[j]}\n\n{stage_with_prompt}"
                        else:
                            batch_conversation_history[j] += f"\n\n{stage_with_prompt}"
                            
                    except Exception as single_e:
                        logger.error(f"    问题 {j} 阶段 {stage_idx + 1} 单个生成也失败: {single_e}")
                        batch_stage_responses[j].append("")
                        batch_stage_entropy_histories[j].append([])
                        batch_stage_minp_histories[j].append([])
        
        # 打印当前批次结果统计
        batch_num = i//batch_size + 1
        total_batches = (len(questions) + batch_size - 1)//batch_size
        logger.info(f"\n=== 批次 {batch_num}/{total_batches} 完成统计 ===")
        
        # 统计各阶段响应情况
        stage_stats = [0, 0, 0, 0]  # 每个阶段非空响应的数量
        for j in range(current_batch_size):
            for stage_idx in range(4):
                if j < len(batch_stage_responses) and stage_idx < len(batch_stage_responses[j]):
                    if batch_stage_responses[j][stage_idx].strip():
                        stage_stats[stage_idx] += 1
        
        logger.info(f"批次大小: {current_batch_size}")
        for stage_idx in range(4):
            success_rate = (stage_stats[stage_idx] / current_batch_size) * 100
            logger.info(f"阶段{stage_idx+1}成功生成: {stage_stats[stage_idx]}/{current_batch_size} ({success_rate:.1f}%)")
        
        # 统计合并后响应情况
        combined_success = 0
        for j in range(current_batch_size):
            if j < len(batch_stage_responses):
                combined_response = " ".join(batch_stage_responses[j])
                if combined_response.strip():
                    combined_success += 1
        
        combined_success_rate = (combined_success / current_batch_size) * 100
        logger.info(f"合并响应成功: {combined_success}/{current_batch_size} ({combined_success_rate:.1f}%)")
        
        # 显示几个样本示例
        logger.info(f"\n--- 批次 {batch_num} 样本示例 ---")
        for j in range(min(3, current_batch_size)):  # 显示前3个样本
            logger.info(f"样本 {j+1}:")
            for stage_idx in range(4):
                if j < len(batch_stage_responses) and stage_idx < len(batch_stage_responses[j]):
                    response_with_prompt = batch_stage_responses[j][stage_idx]
                    if response_with_prompt.strip():
                        # 提取生成的部分（去掉提示词前缀）
                        stage_prompts_template = get_stage_prompts(config, "", dataset_type)
                        prompt_prefix = stage_prompts_template[stage_idx]
                        generated_part = extract_generated_content(response_with_prompt, prompt_prefix)
                        logger.info(f"  阶段{stage_idx+1}生成内容: {generated_part}")
                        logger.info(f"  阶段{stage_idx+1}完整响应: {response_with_prompt}")
                    else:
                        logger.info(f"  阶段{stage_idx+1}生成内容: [空]")
        
        # 将当前批次结果添加到总结果中
        all_stage_responses.extend(batch_stage_responses)
        all_stage_entropy_histories.extend(batch_stage_entropy_histories)
        all_stage_minp_histories.extend(batch_stage_minp_histories)
    
    logger.info(f"批量4阶段生成完成")
    return all_stage_responses, all_stage_entropy_histories, all_stage_minp_histories

def batch_generate(model, tokenizer, questions: List[str], dynamic_min_p_processor, generation_config: Dict, batch_size: int = 4) -> Tuple[List[str], List[List[float]], List[List[float]]]:
    """
    批量生成文本
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        questions: 问题列表
        dynamic_min_p_processor: 动态min_p处理器
        generation_config: 生成配置
        batch_size: 批量大小
        
    Returns:
        Tuple[List[str], List[List[float]], List[List[float]]]: (生成的响应列表, 每个样本的熵值历史列表, 每个样本的min_p历史列表)
    """
    all_responses = []
    all_entropy_histories = []
    all_minp_histories = []
    
    # 分批处理
    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i:i + batch_size]
        current_batch_size = len(batch_questions)
        
        # 获取logger，如果没有则使用基本输出
        logger = logging.getLogger("dynamics_minp")
        if logger.handlers:
            logger.info(f"正在处理批次 {i//batch_size + 1}/{(len(questions) + batch_size - 1)//batch_size}，包含 {current_batch_size} 个样本")
        else:
            print(f"正在处理批次 {i//batch_size + 1}/{(len(questions) + batch_size - 1)//batch_size}，包含 {current_batch_size} 个样本")
        
        # 重置熵记录
        dynamic_min_p_processor.reset_entropy_history()
        
        # 使用padding确保批次中所有序列长度相同
        inputs = tokenizer(
            batch_questions, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        ).to(model.device)
        
        # 批量生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=generation_config['max_new_tokens'],
                logits_processor=LogitsProcessorList([dynamic_min_p_processor]),
                do_sample=generation_config['do_sample'],
                temperature=generation_config['temperature'],
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 解码批量结果
        batch_responses = []
        for j, output in enumerate(outputs):
            # 解码生成的文本
            generated_text = tokenizer.decode(output, skip_special_tokens=True)
            # 移除输入部分，只保留生成的部分
            original_question = batch_questions[j]
            if generated_text.startswith(original_question):
                response = generated_text[len(original_question):].strip()
            else:
                # 如果无法精确匹配，尝试找到问题结束位置
                response = generated_text.strip()
            
            batch_responses.append(response)
        
        # 获取每个样本的熵值和min_p历史
        batch_entropy_histories = dynamic_min_p_processor.get_batch_entropy_histories()
        batch_minp_histories = dynamic_min_p_processor.get_batch_minp_histories()
        batch_entropy_list = []
        batch_minp_list = []
        for j in range(current_batch_size):
            entropy_history = batch_entropy_histories.get(j, [])
            minp_history = batch_minp_histories.get(j, [])
            batch_entropy_list.append(entropy_history)
            batch_minp_list.append(minp_history)
        
        # 打印当前批次结果统计
        batch_num = i//batch_size + 1
        total_batches = (len(questions) + batch_size - 1)//batch_size
        logger.info(f"\n=== 普通批次 {batch_num}/{total_batches} 完成统计 ===")
        
        # 统计响应情况
        successful_responses = sum(1 for response in batch_responses if response.strip())
        success_rate = (successful_responses / current_batch_size) * 100
        
        logger.info(f"批次大小: {current_batch_size}")
        logger.info(f"成功生成响应: {successful_responses}/{current_batch_size} ({success_rate:.1f}%)")
        
        # 显示几个样本示例
        logger.info(f"\n--- 普通批次 {batch_num} 样本示例 ---")
        for j in range(min(3, current_batch_size)):  # 显示前3个样本
            response = batch_responses[j] if j < len(batch_responses) else ""
            logger.info(f"样本 {j+1}: {response}")
        
        all_responses.extend(batch_responses)
        all_entropy_histories.extend(batch_entropy_list)
        all_minp_histories.extend(batch_minp_list)
            
    
    return all_responses, all_entropy_histories, all_minp_histories


# --- 核心组件 3: 端到端使用示例 ---
def main(config_path: str = "/cephfs/shared/sunyifan/Min-p-CoT/dynamics_minp_config.json"):
    """
    一个完整的示例，展示如何加载模型并使用我们的动态 min_p 策略。
    所有参数都从配置文件中读取。
    
    Args:
        config_path: 配置文件路径
    """
    # 加载配置文件
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"配置加载失败: {e}")
        return
    
    # 初始化日志系统
    output_dir = config['evaluation']['output_dir']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"dynamics_minp_{config['model']['model_name']}_{timestamp}"
    logger = setup_logging(output_dir, experiment_name)
    
    logger.info("=== 配置信息 ===")
    logger.info(f"实验名称: {experiment_name}")
    logger.info(f"配置文件: {config_path}")
    logger.info(f"模型配置: {config['model']}")
    logger.info(f"数据集配置: {config['dataset']}")
    logger.info(f"动态min_p配置: {config['dynamic_minp']}")
    logger.info(f"生成配置: {config['generation']}")
    logger.info("=" * 50)
    
    # 确定要使用的本地模型路径
    model_name = config['model']['model_name']
    model_path = f"/cephfs/shared/sunyifan/Model/{model_name}"
    
    logger.info(f"正在加载本地模型: {model_path}")
    
    # 验证本地模型是否存在
    if not verify_local_model(model_path):
        logger.error(f"错误: 本地模型不存在或不完整: {model_path}")
        logger.error("请检查模型文件是否正确下载到指定目录")
        logger.error("必需文件: config.json, pytorch_model.bin 或 model.safetensors")
        return
    
    # 加载模型和分词器
    try:
        logger.info("正在加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info("✓ Tokenizer加载成功")
        
        # 确保tokenizer有padding token（批量生成必需）
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("✓ 设置padding token为eos_token")
        
        logger.info("正在加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=config['model']['device_map'],
            torch_dtype=get_torch_dtype(config['model']['torch_dtype'])
        )
        logger.info("✓ 模型加载完毕")
        logger.info(f"模型设备: {next(model.parameters()).device}")
        
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        logger.error(f"尝试的路径: {model_path}")
        return

    # 从配置文件读取动态策略超参数
    dynamic_minp_config = config['dynamic_minp']
    H_MIN = dynamic_minp_config['h_min']
    H_MAX = dynamic_minp_config['h_max']
    MIN_P_MIN = dynamic_minp_config['min_p_min']
    MIN_P_MAX = dynamic_minp_config['min_p_max']
    
    # 滑动窗口相关参数
    USE_SLIDING_WINDOW = dynamic_minp_config.get('use_sliding_window', False)
    WINDOW_SIZE = dynamic_minp_config.get('window_size', 10)
    PERCENTILE = dynamic_minp_config.get('percentile', 95)
    MIN_SAMPLES_FOR_UPDATE = dynamic_minp_config.get('min_samples_for_update', 5)
    
    # 新增：熵阈值和固定min_p参数
    ENTROPY_THRESHOLD = dynamic_minp_config.get('entropy_threshold', None)
    FIXED_LOW_MINP = dynamic_minp_config.get('fixed_low_minp', 0.2)

    # 实例化我们的动态 min_p 控制器
    dynamic_min_p_processor = EntropyDynamicMinPLogitsProcessor(
        h_min=H_MIN,
        h_max=H_MAX,
        min_p_min=MIN_P_MIN,
        min_p_max=MIN_P_MAX,
        use_sliding_window=USE_SLIDING_WINDOW,
        window_size=WINDOW_SIZE,
        percentile=PERCENTILE,
        min_samples_for_update=MIN_SAMPLES_FOR_UPDATE,
        entropy_threshold=ENTROPY_THRESHOLD,
        fixed_low_minp=FIXED_LOW_MINP
    )
    
    logger.info(f"滑动窗口模式: {'启用' if USE_SLIDING_WINDOW else '禁用'}")
    if USE_SLIDING_WINDOW:
        logger.info(f"窗口大小: {WINDOW_SIZE}, 百分位数: {PERCENTILE}%, 最小更新样本数: {MIN_SAMPLES_FOR_UPDATE}")
    
    logger.info(f"动态min_p策略: 高熵(>{ENTROPY_THRESHOLD if ENTROPY_THRESHOLD else 'dynamic'})使用线性映射[{MIN_P_MIN}, {MIN_P_MAX}], 低熵使用固定值{FIXED_LOW_MINP}")

    # 准备数据集和结果记录
    dataset_config = config['dataset']
    use_custom_prompts = dataset_config['use_custom_prompts']
    
    results = []
    dataset_items = []
    
    logger.info("正在加载数据集...")
    dataset_path = "/cephfs/shared/sunyifan/Dataset/"+f"{dataset_config['dataset_name']}" + "/test.jsonl"
    dataset_items = load_local_dataset(dataset_path)
    logger.info(f"✓ 数据集加载完成，共 {len(dataset_items)} 个样本")
    logger.info(f"数据集路径: {dataset_path}")
    
    # 获取生成配置
    generation_config = config['generation']
    enable_batch = generation_config.get('enable_batch_generation', False)
    batch_size = generation_config.get('batch_size', 4)
    
    # 提取问题列表并增强prompt
    cot_method = config.get('generation', {}).get('cot_method', 'simple')
    # 统一处理4阶段CoT的不同命名方式
    if cot_method == '4_stage_cot':
        cot_method = '4_stage'
    use_4stage_cot = (cot_method == '4_stage') or config.get('generation', {}).get('use_4stage_cot', False)
    
    cot_mode_map = {
        'simple': '简单CoT',
        '4_stage': '4阶段CoT',
        'one_shot': 'One-shot学习',
        'none': '无CoT'
    }
    cot_mode = cot_mode_map.get(cot_method, '简单CoT')
    
    logger.info(f"CoT方法: {cot_method} ({cot_mode})")
    
    questions = []
    dataset_type = dataset_config.get('dataset_type', 'math')
    
    # 检查是否使用one-shot示例
    use_one_shot = config.get('one_shot', False)
    one_shot_prompts = config.get('one_shot_prompts', '') if use_one_shot else ""
    
    # 构建问题列表，如果启用one-shot，则在每个问题前添加示例
    logger.info(f"正在构建问题列表...")
    if use_one_shot:
        logger.info(f"✓ 启用One-shot学习，将在每个问题前添加示例")
    
    for item in dataset_items:
        base_question = item["question"]
        
        if cot_method == 'none':
            # 不使用CoT，直接使用原始问题
            final_question = base_question
        elif cot_method == '4_stage':
            # 4阶段CoT将在生成时处理，这里使用原始问题
            final_question = base_question
        else:
            # 简单CoT
            final_question = enhance_prompt_with_cot(config, base_question, dataset_type, cot_method)
        
        # 如果启用one-shot，在问题前添加示例
        if use_one_shot and one_shot_prompts:
            final_question = f"{one_shot_prompts}\n\nNow solve this problem:\n\nQuestion: {final_question}\n\nAnswer:"
        
        questions.append(final_question)
    
    logger.info(f"✓ 完成 {len(questions)} 个问题的构建 ({cot_mode}{'，包含One-shot示例' if use_one_shot else ''})")
    
    logger.info("\n--- 开始生成和评估 ---")
    logger.info(f"总样本数: {len(dataset_items)}")
    logger.info(f"批量生成: {'启用' if enable_batch else '禁用'}")
    if enable_batch:
        logger.info(f"批量大小: {batch_size}")
    
    # 获取试验配置
    evaluation_config = config.get('evaluation', {})
    num_trials = evaluation_config.get('num_trials', 1)
    average_results = evaluation_config.get('average_results', True)
    
    logger.info(f"试验次数: {num_trials}")
    if num_trials > 1 and average_results:
        logger.info("将计算多次试验的平均结果")
    
    # 记录总体开始时间
    total_start_time = time.time()
    
    # 运行多次试验
    trials_results = run_multiple_trials(
        model, tokenizer, questions, dataset_items, dataset_type, 
        dynamic_min_p_processor, generation_config, config, logger, num_trials
    )
    
    # 记录总体结束时间
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    # 处理试验结果
    logger.info("\n--- 试验结果统计 ---")
    
    # 获取最终结果
    if num_trials > 1:
        logger.info(f"多次试验结果 (共{num_trials}次):")
        logger.info(f"平均准确率: {trials_results['average_accuracy']:.4f} ± {trials_results['accuracy_std']:.4f}")
        logger.info(f"各次试验准确率: {[f'{acc:.4f}' for acc in trials_results['individual_accuracies']]}")
        logger.info(f"平均生成时间: {sum(trials_results['average_generation_times'])/len(trials_results['average_generation_times']):.2f}秒/样本")
        
        # 使用第一次试验的结果作为详细结果展示
        main_trial = trials_results['all_trials'][0]
    else:
        main_trial = trials_results['all_trials'][0]
        logger.info(f"单次试验结果:")
        logger.info(f"准确率: {main_trial['accuracy']:.4f}")
        logger.info(f"平均生成时间: {sum(main_trial['generation_times'])/len(main_trial['generation_times']):.2f}秒/样本")
    
    # 构建详细结果用于保存
    results = []
    for i, item in enumerate(dataset_items):
        if cot_method == '4_stage':
            # 4阶段CoT结果
            result = {
                "sample_id": i,
                "question": item["question"],
                "ground_truth": item.get("answer", ""),
                "generation_mode": "4_stage_cot",
                "stage_responses": main_trial['stage_responses'][i] if i < len(main_trial['stage_responses']) else ["", "", "", ""],
                "full_response": main_trial['responses'][i] if i < len(main_trial['responses']) else "",
                "predicted_answer": main_trial['predicted_answers'][i] if i < len(main_trial['predicted_answers']) else "",
                "dataset_type": item.get("dataset_type", dataset_type),
                "level": item.get("level", ""),
                "category": item.get("category", ""),
                "generation_time": main_trial['generation_times'][i] if i < len(main_trial['generation_times']) else 0,
                "stage_entropy_histories": main_trial['stage_entropy_histories'][i] if i < len(main_trial['stage_entropy_histories']) else [[], [], [], []],
                "stage_minp_histories": main_trial['stage_minp_histories'][i] if i < len(main_trial['stage_minp_histories']) else [[], [], [], []],
                "combined_entropy_history": main_trial['entropy_histories'][i] if i < len(main_trial['entropy_histories']) else [],
                "combined_minp_history": main_trial['minp_histories'][i] if i < len(main_trial['minp_histories']) else []
            }
        else:
            # 非4阶段结果
            result = {
                "sample_id": i,
                "question": item["question"],
                "ground_truth": item.get("answer", ""),
                "generation_mode": cot_method,
                "generated_response": main_trial['responses'][i] if i < len(main_trial['responses']) else "",
                "predicted_answer": main_trial['predicted_answers'][i] if i < len(main_trial['predicted_answers']) else "",
                "dataset_type": item.get("dataset_type", dataset_type),
                "level": item.get("level", ""),
                "category": item.get("category", ""),
                "generation_time": main_trial['generation_times'][i] if i < len(main_trial['generation_times']) else 0,
                "entropy_history": main_trial['entropy_histories'][i] if i < len(main_trial['entropy_histories']) else [],
                "minp_history": main_trial['minp_histories'][i] if i < len(main_trial['minp_histories']) else []
            }
        
        # 添加统计信息
        entropy_hist = result.get('combined_entropy_history', result.get('entropy_history', []))
        minp_hist = result.get('combined_minp_history', result.get('minp_history', []))
        
        result["entropy_stats"] = {
            "mean": sum(entropy_hist) / len(entropy_hist) if entropy_hist else 0,
            "max": max(entropy_hist) if entropy_hist else 0,
            "min": min(entropy_hist) if entropy_hist else 0,
            "count": len(entropy_hist)
        }
        result["minp_stats"] = {
            "mean": sum(minp_hist) / len(minp_hist) if minp_hist else 0,
            "max": max(minp_hist) if minp_hist else 0,
            "min": min(minp_hist) if minp_hist else 0,
            "count": len(minp_hist)
        }
        
        # 添加正确性判断
        if item.get("answer") and i < len(main_trial['predicted_answers']):
            pred_answer = main_trial['predicted_answers'][i]
            ground_truth = item["answer"]
            
            try:
                pred_clean = str(pred_answer).strip().lower()
                truth_clean = str(ground_truth).strip().lower()
                
                # 数值比较
                try:
                    pred_num = float(pred_clean)
                    truth_num = float(truth_clean)
                    is_correct = abs(pred_num - truth_num) < 1e-6
                except (ValueError, TypeError):
                    # 字符串比较
                    is_correct = pred_clean == truth_clean
                
                result["is_correct"] = is_correct
            except Exception as e:
                logger.error(f"样本 {i} 正确性判断失败: {e}")
                result["is_correct"] = False
        
        # 如果是多次试验，添加所有试验的结果
        if num_trials > 1:
            result["all_trials"] = {
                "predicted_answers": [trial['predicted_answers'][i] if i < len(trial['predicted_answers']) else "" for trial in trials_results['all_trials']],
                "accuracies": [trial['accuracy'] for trial in trials_results['all_trials']],
                "generation_times": [trial['generation_times'][i] if i < len(trial['generation_times']) else 0 for trial in trials_results['all_trials']]
            }
        
        results.append(result)
    
    # 添加试验汇总信息
    experiment_summary = {
        "experiment_config": {
            "model_name": config['model']['model_name'],
            "dataset_name": dataset_config['dataset_name'],
            "cot_method": cot_method,
            "use_one_shot": use_one_shot,
            "num_trials": num_trials,
            "batch_size": batch_size if enable_batch else 1,
            "enable_batch_generation": enable_batch
        },
        "results_summary": {
            "total_samples": len(dataset_items),
            "total_time": total_time,
            "average_time_per_sample": total_time / len(dataset_items) if dataset_items else 0
        }
    }
    
    if num_trials > 1:
        experiment_summary["results_summary"].update({
            "average_accuracy": trials_results['average_accuracy'],
            "accuracy_std": trials_results['accuracy_std'],
            "individual_accuracies": trials_results['individual_accuracies']
        })
    else:
        experiment_summary["results_summary"]["accuracy"] = main_trial['accuracy']
    
    # 保存结果
    if config.get('evaluation', {}).get('save_results', True):
        output_dir = config.get('evaluation', {}).get('output_dir', './output')
        
        # 构建文件名（不包含时间戳）
        model_name = config['model']['model_name'].replace('/', '_')
        dataset_name = dataset_config['dataset_name']
        
        # 获取投票配置信息
        voting_config = config.get('voting', {})
        enable_voting = voting_config.get('enable_voting', False)
        num_votes = voting_config.get('num_votes', 5)
        confidence_threshold = voting_config.get('confidence_threshold', 0.0)
        
        filename_parts = [
            f"results_{model_name}_{dataset_name}_{cot_method}",
            f"trials{num_trials}" if num_trials > 1 else "single"
        ]
        
        # 添加投票信息
        if enable_voting:
            filename_parts.append(f"vote{num_votes}")
            if confidence_threshold > 0:
                # 将置信度格式化为整数（如0.6变成60）
                confidence_percent = int(confidence_threshold * 100)
                filename_parts.append(f"conf{confidence_percent}")
        else:
            filename_parts.append("novote")
        
        # 添加其他信息
        filename_parts.append(f"batch{batch_size}" if enable_batch else "sequential")
        filename_parts.append("oneshot" if use_one_shot else "no_oneshot")
        
        filename = "_".join(filename_parts) + ".json"
        
        output_path = os.path.join(output_dir, filename)
        
        final_results = {
            "experiment_summary": experiment_summary,
            "detailed_results": results
        }
        
        try:
            save_results(final_results, output_path)
            logger.info(f"结果已保存到: {output_path}")
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
    
    logger.info("\n实验完成!")
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="动态Min-P文本生成实验")
    parser.add_argument("--config", type=str, default="/cephfs/shared/sunyifan/Min-p-CoT/dynamics_minp_config.json", 
                      help="配置文件路径")
    
    args = parser.parse_args()
    
    try:
        results = main(args.config)
        print(f"\n实验完成，共处理 {len(results)} 个样本")
    except Exception as e:
        print(f"实验失败: {e}")
        import traceback
        traceback.print_exc()
            