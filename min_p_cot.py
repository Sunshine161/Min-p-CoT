import torch
import asyncio
import numpy as np
import re
from datetime import datetime
from typing import List, Dict, Optional, Iterator, AsyncIterator, Callable, Tuple, Any
import logging
import random
import json
import sys
from logger import setup_logger
from pathlib import Path
from extractor import extract_answer

# 强制要求vLLM，不提供模拟模式
from vllm import LLM, SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils import random_uuid
from vllm.outputs import RequestOutput

# 设置基础路径
BASE_DIR = Path(__file__).parent
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

def load_config(config_path: str = "config.json", logger: Optional['Logger'] = None) -> Dict[str, Any]:
    """
    加载JSON配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        Dict: 配置字典
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件未找到: {config_file}")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except json.JSONDecodeError as e:
        raise ValueError(f"配置文件JSON格式错误: {e}")
    except Exception as e:
        raise RuntimeError(f"加载配置文件失败: {e}")

def validate_config(config: Dict[str, Any]) -> bool:
    """
    验证配置文件的完整性和正确性
    
    Args:
        config: 配置字典
        
    Returns:
        bool: 验证是否通过
    """
    required_sections = [
        "model_config", "dataset_config", "generation_config", 
        "cot_config", "experiment_config", "prompt_templates"
    ]
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"配置文件缺少必需的节: {section}")
    
    # 验证step_minp_values长度
    step_minp = config["cot_config"]["step_minp_values"]
    if len(step_minp) != 4:
        raise ValueError("step_minp_values必须包含4个值")
    
    # 验证min_p值范围
    for i, minp in enumerate(step_minp):
        if not 0.0 <= minp <= 1.0:
            raise ValueError(f"step_minp_values[{i}] = {minp} 不在有效范围[0.0, 1.0]内")
    
    # 验证step_guidance（如果存在）
    if "step_guidance" in config["cot_config"]:
        step_guidance = config["cot_config"]["step_guidance"]
        if not isinstance(step_guidance, list):
            raise ValueError("step_guidance必须是一个列表")
        if len(step_guidance) != 4:
            raise ValueError(f"step_guidance必须包含4个引导词，当前有{len(step_guidance)}个")
        for i, guidance in enumerate(step_guidance):
            if not isinstance(guidance, str):
                raise ValueError(f"step_guidance[{i}]必须是字符串类型")
    
    # 验证voting_config（如果存在）
    if "voting_config" in config:
        voting_config = config["voting_config"]
        if voting_config.get("enabled", False):
            num_paths = voting_config.get("num_paths", 1)
            if not isinstance(num_paths, int) or num_paths < 1:
                raise ValueError("num_paths必须是大于0的整数")
            
            voting_method = voting_config.get("voting_method", "majority")
            if voting_method not in ["majority", "confidence"]:
                raise ValueError("voting_method必须是'majority'或'confidence'")
            
            confidence_threshold = voting_config.get("confidence_threshold", 0.5)
            if not 0.0 <= confidence_threshold <= 1.0:
                raise ValueError("confidence_threshold必须在[0.0, 1.0]范围内")
            
            batch_generation = voting_config.get("batch_generation", True)
            if not isinstance(batch_generation, bool):
                raise ValueError("batch_generation必须是布尔值")
    
    temp_logger = logging.getLogger("Config")
    temp_logger.info("配置文件验证通过")
    return True

def set_random_seed(seed: int, logger: Optional['Logger'] = None):
    """
    设置随机种子以确保结果可复现
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    # 设置torch的随机种子
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # 确保CUDA操作的确定性
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    logger.info(f"随机种子已设置为: {seed}")



def load_dataset(dataset_path: str, shuffle: bool = True, random_seed: Optional[int] = None, logger: Optional['Logger'] = None):
    """
    加载数据集
    
    Args:
        dataset_path: 数据集路径
        shuffle: 是否打乱数据集顺序
        random_seed: 随机种子
    """
    if random_seed is not None:
        set_random_seed(random_seed, logger)
    
    dataset = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            dataset.append({"problem": data["problem"], "solution": data["solution"]})
    
    if shuffle:
        random.shuffle(dataset)
    
    logger.info(f"数据集加载完成，共 {len(dataset)} 条数据{'，已打乱顺序' if shuffle else ''}")
    return dataset


def generate_cot(
    model, 
    prompts: List[str], 
    config: Dict[str, Any],
    logger: Optional['Logger'] = None,
):
    """
    通过四步骤引导和vLLM输出生成CoT。
    在每个步骤开始时添加引导词，当检测到下一步骤标识符时自动切换min_p值。

    Args:
        model: vLLM模型实例
        problem: 初始问题
        config: 配置字典
        logger: 日志记录器

    Yields:
        str: 流式生成的文本块。
    """
    if logger is None:
        logger = Logger()
    
    # 从配置中提取参数
    cot_config = config["cot_config"]
    gen_config = config["generation_config"]
    prompt_config = config["prompt_templates"]
    
    cot_type = cot_config["cot_type"]
    step_minp = cot_config["step_minp_values"]
    max_tokens = gen_config["max_tokens"]
    temperature = gen_config["temperature"]
    stop_sequences = gen_config["stop_sequences"]

    if cot_type == "No-CoT":
        for prompt in prompts:
            prompt += "Please present the final answer in \\boxed{*}.\n\n"
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=len(step_minp)*max_tokens,
            min_p=max(step_minp),
            stop=stop_sequences
        )
        outputs = model.generate(prompts, sampling_params)
        generated_texts = [output.outputs[0].text for output in outputs]
        return generated_texts

    step_guidance = cot_config["step_guidance"]
    
    # 步骤检测配置
    step_detection = cot_config["step_detection"]
    use_regex = step_detection["use_regex"]
    

    # 构建初始prompt
    guidance_prompt = prompt_config["guidance_prompt"]
    full_prompts = []
    for prompt in prompts:
        full_prompts.append(guidance_prompt + "\n\nQuestion: " + prompt)
    

    current_step = 0
    total_generated_texts = full_prompts
    problem_length = [len(prompt) for prompt in full_prompts]
    
    # 根据配置选择检测模式
    if use_regex:
        step_patterns = step_detection["regex_patterns"]
    else:
        step_patterns = step_detection["patterns"]
    
    # 打印步骤描述和引导词
    step_descriptions = cot_config["step_descriptions"]
    logger.info(f"开始CoT生成，共4个步骤:")
    
    while current_step < len(step_minp):
        total_generated_texts = [total_generated_text + f"\n\nStep {current_step+1}:{step_guidance[current_step]}" for total_generated_text in total_generated_texts]
        
        # 配置当前步骤的采样参数
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            min_p=step_minp[current_step],
            stop=stop_sequences
        )
        
        logger.info(f"开始CoT步骤 {current_step+1}，min_p={step_minp[current_step]}")
        
        outputs = model.generate(total_generated_texts, sampling_params)
        
        if not outputs or not outputs[0].outputs:
            logger.warning(f"步骤 {current_step+1} 生成失败")
            break
            
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text

            # 检查是否包含下一步骤的标识符
            step_detected = False
            if current_step < 3 and step_detection["enabled"]:
                pattern = step_patterns[current_step]
                
                if use_regex:
                    # 使用正则表达式检测
                    match = re.search(pattern, generated_text)
                    step_found = match is not None
                    step_pos = match.start() if match else -1
                else:
                    # 使用简单字符串查找
                    step_pos = generated_text.find(pattern)
                    step_found = step_pos != -1
                
                if step_found:
                    # 检测到下一个步骤标识符
                    step_detected = True
                    
                    # 只输出到步骤标识符之前的内容
                    text_before_step = generated_text[:step_pos]
                    
                    
                    # 更新total_generated_text到步骤标识符位置
                    total_generated_texts[i] += text_before_step
                    
                    
            # 如果没有检测到步骤切换，输出完整的生成文本
            if not step_detected:
                total_generated_texts[i] += generated_text

        current_step += 1
    logger.info(f"CoT生成完成，共 {len(total_generated_texts)} 条CoT")
    
    return [total_generated_text[problem_length[i]:] for i, total_generated_text in enumerate(total_generated_texts)]


def vote_for_best_answer(
    generated_paths: List[str], 
    config: Dict[str, Any], 
    dataset_type: str = "math",
    logger: Optional['Logger'] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    从多条生成路径中投票选择最佳答案
    
    Args:
        generated_paths: 生成的多条路径列表
        config: 配置字典
        dataset_type: 数据集类型
        logger: 日志记录器
        
    Returns:
        Tuple[str, Dict]: 选中的最佳路径和投票统计信息
    """
    if not generated_paths:
        return "", {"error": "没有生成路径"}
    
    if len(generated_paths) == 1:
        return generated_paths[0], {"single_path": True}
    
    voting_config = config.get("voting_config", {})
    voting_method = voting_config.get("voting_method", "majority")
    
    # 从每条路径中提取答案
    extracted_answers = []
    answer_to_paths = {}
    
    for i, path in enumerate(generated_paths):
        try:
            answer = extract_answer(path, dataset_type=dataset_type)
            extracted_answers.append(answer)
            
            # 记录答案对应的路径
            if answer not in answer_to_paths:
                answer_to_paths[answer] = []
            answer_to_paths[answer].append(i)
            
        except Exception as e:
            if logger:
                logger.warning(f"路径 {i} 答案提取失败: {e}")
            extracted_answers.append(None)
    
    # 统计答案频次
    answer_counts = {}
    valid_answers = [ans for ans in extracted_answers if ans is not None]
    
    for answer in valid_answers:
        answer_counts[answer] = answer_counts.get(answer, 0) + 1
    
    if not answer_counts:
        # 如果没有有效答案，返回第一条路径
        return generated_paths[0], {"error": "没有有效答案", "fallback_to_first": True}
    
    # 投票选择
    if voting_method == "majority":
        # 多数投票：选择出现次数最多的答案
        best_answer = max(answer_counts.keys(), key=lambda x: answer_counts[x])
        best_count = answer_counts[best_answer]
        
        # 选择该答案对应的第一条路径
        best_path_idx = answer_to_paths[best_answer][0]
        best_path = generated_paths[best_path_idx]
        
        voting_stats = {
            "voting_method": "majority",
            "best_answer": best_answer,
            "vote_count": best_count,
            "total_paths": len(generated_paths),
            "valid_paths": len(valid_answers),
            "confidence": best_count / len(valid_answers) if valid_answers else 0,
            "answer_distribution": answer_counts,
            "selected_path_idx": best_path_idx
        }
        
    else:  # confidence-based voting
        # 基于置信度的投票（这里简化为选择最长的推理路径）
        path_scores = []
        for i, path in enumerate(generated_paths):
            if extracted_answers[i] is not None:
                # 简单的置信度评估：路径长度和答案频次的结合
                length_score = len(path) / 1000  # 归一化长度分数
                frequency_score = answer_counts.get(extracted_answers[i], 0) / len(valid_answers)
                combined_score = 0.3 * length_score + 0.7 * frequency_score
                path_scores.append((i, combined_score))
        
        if path_scores:
            best_path_idx, best_score = max(path_scores, key=lambda x: x[1])
            best_path = generated_paths[best_path_idx]
            best_answer = extracted_answers[best_path_idx]
            
            voting_stats = {
                "voting_method": "confidence",
                "best_answer": best_answer,
                "confidence_score": best_score,
                "total_paths": len(generated_paths),
                "valid_paths": len(valid_answers),
                "answer_distribution": answer_counts,
                "selected_path_idx": best_path_idx
            }
        else:
            # 回退到第一条路径
            best_path = generated_paths[0]
            voting_stats = {"error": "置信度计算失败", "fallback_to_first": True}
    
    if logger:
        logger.info(f"投票结果: 选择路径 {voting_stats.get('selected_path_idx', 0)}, "
                   f"答案: {voting_stats.get('best_answer', 'N/A')}, "
                   f"置信度: {voting_stats.get('confidence', voting_stats.get('confidence_score', 'N/A'))}")
    
    return best_path, voting_stats


def generate_multiple_paths(
    model, 
    prompts: List[str], 
    config: Dict[str, Any],
    logger: Optional['Logger'] = None,
) -> List[List[str]]:
    """
    为每个prompt批量生成多条路径
    
    Args:
        model: vLLM模型实例
        prompts: 问题列表
        config: 配置字典
        logger: 日志记录器
        
    Returns:
        List[List[str]]: 每个prompt对应的多条路径列表
    """
    voting_config = config.get("voting_config", {})
    num_paths = voting_config.get("num_paths", 1)
    
    if not voting_config.get("enabled", False) or num_paths == 1:
        # 如果未启用投票或只生成1条路径，使用原来的方法
        single_paths = generate_cot(model, prompts, config, logger)
        return [[path] for path in single_paths]
    
    if logger:
        logger.info(f"批量生成模式：为 {len(prompts)} 个问题各生成 {num_paths} 条路径")
    
    # 准备批量输入：每个prompt重复num_paths次
    batch_prompts = []
    prompt_indices = []  # 记录每个批量prompt对应的原始prompt索引
    
    for i, prompt in enumerate(prompts):
        for path_idx in range(num_paths):
            batch_prompts.append(prompt)
            prompt_indices.append(i)
    
    # 批量生成所有路径
    all_generated_paths = []
    
    # 为了增加多样性，我们可以运行多次批量生成
    for run_idx in range(num_paths):
        try:
            # 批量生成当前轮次的所有路径
            current_batch_prompts = [prompts[i] for i in range(len(prompts))]
            generated_paths = generate_cot(model, current_batch_prompts, config, logger)
            all_generated_paths.extend(generated_paths)
            
        except Exception as e:
            if logger:
                logger.error(f"第 {run_idx + 1} 次批量生成失败: {e}")
            # 如果失败，使用默认配置重试
            try:
                current_batch_prompts = [prompts[i] for i in range(len(prompts))]
                generated_paths = generate_cot(model, current_batch_prompts, config, logger)
                all_generated_paths.extend(generated_paths)
            except Exception as e2:
                if logger:
                    logger.error(f"第 {run_idx + 1} 次批量生成重试也失败: {e2}")
                # 如果重试也失败，为每个prompt添加空字符串
                all_generated_paths.extend([""] * len(prompts))
    
    # 重新组织结果：将生成的路径按原始prompt分组
    organized_paths = [[] for _ in range(len(prompts))]
    
    for run_idx in range(num_paths):
        start_idx = run_idx * len(prompts)
        end_idx = start_idx + len(prompts)
        
        for i, path in enumerate(all_generated_paths[start_idx:end_idx]):
            if i < len(organized_paths):
                organized_paths[i].append(path)
    
    # 确保每个prompt都有指定数量的路径
    for i in range(len(organized_paths)):
        while len(organized_paths[i]) < num_paths:
            # 如果路径不足，复制第一条路径
            if organized_paths[i]:
                organized_paths[i].append(organized_paths[i][0])
            else:
                organized_paths[i].append("")
    
    if logger:
        logger.info(f"批量生成完成：共生成 {len(organized_paths)} 组路径，每组 {num_paths} 条")
    
    return organized_paths


def save_results(results: List[Dict], config: Dict[str, Any], output_dir: str, logger: Optional['Logger'] = None):
    """
    保存实验结果
    
    Args:
        results: 结果列表
        config: 配置字典
        output_dir: 输出目录
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    min_p_str = '_'.join(map(str, config['cot_config']['step_minp_values']))
    
    # 安全处理模型名称和数据集名称，避免路径问题
    model_name_safe = config['model_config']['model_name'].replace('/', '_')
    dataset_name_safe = config['dataset_config']['dataset_name'].replace('/', '_')
    
    # 添加投票信息到文件名
    voting_suffix = ""
    if config.get("voting_config", {}).get("enabled", False):
        num_paths = config["voting_config"]["num_paths"]
        voting_method = config["voting_config"]["voting_method"]
        voting_suffix = f"_vote{num_paths}_{voting_method}"
    
    # 检查是否是多次运行，如果是则添加运行次数后缀
    run_suffix = ""
    if "run_id" in config.get("experiment_config", {}):
        run_id = config["experiment_config"]["run_id"]
        if run_id > 1:
            run_suffix = f"_run{run_id}"
    
    file_name = f"min_p_cot_results_{config['cot_config']['cot_type']}_{model_name_safe}_{dataset_name_safe}_{min_p_str}{voting_suffix}{run_suffix}.json"
    results_file = output_path / file_name
    
    # 计算投票相关的统计信息
    voting_summary = {}
    if config.get("voting_config", {}).get("enabled", False):
        voting_results = [r for r in results if "voting_stats" in r and r["voting_stats"]]
        if voting_results:
            total_votes = len(voting_results)
            confidence_scores = []
            answer_distributions = []
            
            for r in voting_results:
                stats = r["voting_stats"]
                if "confidence" in stats:
                    confidence_scores.append(stats["confidence"])
                elif "confidence_score" in stats:
                    confidence_scores.append(stats["confidence_score"])
                
                if "answer_distribution" in stats:
                    answer_distributions.append(stats["answer_distribution"])
            
            voting_summary = {
                "voting_enabled": True,
                "num_paths": config["voting_config"]["num_paths"],
                "voting_method": config["voting_config"]["voting_method"],
                "total_voting_samples": total_votes,
                "avg_confidence": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
                "min_confidence": min(confidence_scores) if confidence_scores else 0,
                "max_confidence": max(confidence_scores) if confidence_scores else 0
            }
    else:
        voting_summary = {"voting_enabled": False}

    output_data = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "config": config,
        "results": results,
        "summary": {
            "total_samples": len([r for r in results if "sample_id" in r]),  # 排除统计结果
            "step_minp_values": config["cot_config"]["step_minp_values"],
            "voting_summary": voting_summary
        }
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"结果已保存到: {results_file}")


def run_cot_experiment(config_path: str = "config.json", model=None):
    """
    运行CoT实验
    
    Args:
        config_path: 配置文件路径
        model: 可选的预创建的vLLM模型实例，如果为None则会创建新模型
    """
    
    # 初始化日志
    logger = setup_logger('/cephfs/shared/sunyifan/Min-p-CoT/logs', f"CoT_Experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # 加载和验证配置
    config = load_config(config_path, logger)
    validate_config(config)
    
    # 提取配置
    model_config = config["model_config"]
    dataset_config = config["dataset_config"]
    experiment_config = config["experiment_config"]
    model_name = model_config["model_name"]
    dataset_name = dataset_config["dataset_name"]
    model_path = model_config["model_path"] + model_name
    dataset_path = dataset_config["dataset_path"] + dataset_name + ".jsonl"
    # 设置随机种子
    set_random_seed(experiment_config["random_seed"], logger)
    
    # 创建输出目录
    output_dir = experiment_config["root_dir"] + experiment_config["output_dir"]
    Path(output_dir).mkdir(exist_ok=True)
    
    # 加载模型（仅在model为None时创建新模型）
    if model is None:
        logger.info(f"加载模型: {model_name}")
        model = LLM(
            model=model_path,
            tensor_parallel_size=model_config["tensor_parallel_size"],
            gpu_memory_utilization=model_config["gpu_memory_utilization"],
            max_model_len=model_config["max_model_len"],
            trust_remote_code=model_config["trust_remote_code"]
        )
    else:
        logger.info(f"使用已提供的模型实例: {model_name}")
    
    # 加载数据集
    logger.info(f"加载数据集: {dataset_name}")
    dataset = load_dataset(
        dataset_path, 
        dataset_config["shuffle"], 
        experiment_config["random_seed"],
        logger
    )
    
    # 处理样本
    results = []
    sample_percentage = dataset_config["sample_percentage"]
    num_samples = int(len(dataset) * sample_percentage)
    
    correct_count = 0
    voting_config = config.get("voting_config", {})
    use_voting = voting_config.get("enabled", False)
    
    batch_size = experiment_config["batch_size"]
    for i in range(0, num_samples, batch_size):
        batch_dataset = dataset[:num_samples][i:i+batch_size]
        batch_prompts = [sample["problem"] for sample in batch_dataset]
        batch_solutions = [sample["solution"] for sample in batch_dataset]
        
        if use_voting:
            # 使用投票机制
            logger.info(f"使用投票机制，每个问题生成 {voting_config.get('num_paths', 5)} 条路径")
            all_paths = generate_multiple_paths(
                model=model,
                prompts=batch_prompts,
                config=config,
                logger=logger
            )
            
            # 为每个问题的多条路径进行投票
            generated_cots = []
            voting_stats_list = []
            
            for j, paths_for_prompt in enumerate(all_paths):
                best_path, voting_stats = vote_for_best_answer(
                    paths_for_prompt, 
                    config, 
                    dataset_type=dataset_config["dataset_type"],
                    logger=logger
                )
                generated_cots.append(best_path)
                voting_stats_list.append(voting_stats)
        else:
            # 不使用投票，使用原来的方法
            generated_cots = generate_cot(
                model=model,
                prompts=batch_prompts,
                config=config,
                logger=logger
            )
            voting_stats_list = [{"voting_enabled": False} for _ in generated_cots]
        
        for j, generated_cot in enumerate(generated_cots):
            extracted_answer = extract_answer(generated_cot, dataset_type=dataset_config["dataset_type"])
            reference_answer = extract_answer(batch_solutions[j], dataset_type=dataset_config["dataset_type"], reference_answer=True)
            is_correct = extracted_answer == reference_answer
            
            if is_correct:
                correct_count += 1
        
            # 记录结果
            result = {
                "sample_id": i + j + 1,
                "problem": batch_prompts[j],
                "reference_solution": batch_solutions[j],
                "generated_cot": generated_cot,
                "step_minp_values": config["cot_config"]["step_minp_values"],
                "extracted_answer": extracted_answer,
                "reference_answer": reference_answer,
                "correct": is_correct,
                "voting_stats": voting_stats_list[j] if use_voting else None
            }
            
            # 如果使用投票，还要记录所有生成的路径
            if use_voting and j < len(all_paths):
                result["all_generated_paths"] = all_paths[j]
            
            results.append(result)
        
            # if not is_correct:
            #     logger.info(f"\n{'='*50}")
            #     logger.info(f"样本 {i+j+1}: {batch_prompts[j][:100]}...")
            #     logger.info(f"参考答案: {batch_solutions[j]}")
            #     logger.info(f"提取参考答案: {reference_answer}")
            #     logger.info(f"生成答案: {generated_cot[:200]}...")
            #     logger.info(f"提取答案: {extracted_answer}")
            #     if use_voting:
            #         logger.info(f"投票统计: {voting_stats_list[j]}")
            #     logger.info(f"{'='*50}")
    
    logger.info(f"正确率: {correct_count / num_samples * 100:.2f}%")
    results.append({
        "correct_count": correct_count,
        "total_samples": num_samples,
        "accuracy": correct_count / num_samples * 100
    })
    # 保存结果
    if experiment_config["save_results"]:
        save_results(results, config, output_dir, logger)
    
    logger.info("实验完成!")
    
    # 返回实验结果，供网格搜索使用
    return results


if __name__ == "__main__":
    # 运行实验
    run_cot_experiment(BASE_DIR / "config.json")