#!/usr/bin/env python3
"""
Min-p 网格搜索脚本
用于寻找最佳的 min-p 值组合
"""

import json
import itertools
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Any
import numpy as np

from min_p_cot import run_cot_experiment, load_config, validate_config
from logger import setup_logger

BASE_DIR = Path(__file__).parent

class MinPGridSearch:
    """Min-p 网格搜索类"""
    
    def __init__(self, base_config_path: str, search_config_path: str):
        """
        初始化网格搜索
        
        Args:
            base_config_path: 基础配置文件路径
            output_dir: 搜索结果输出目录
        """
        self.base_config_path = base_config_path
        model_name = load_config(base_config_path)["model_config"]["model_name"]
        dataset_name = load_config(base_config_path)["dataset_config"]["dataset_name"]
        search_interval = self.load_search_config(search_config_path)["search_config"]["search_interval"]
        self.output_dir = Path(self.load_search_config(search_config_path)["search_config"]["output_dir"] + model_name + "_" + dataset_name + "_" + search_interval)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # 设置日志
        self.logger = setup_logger(
            str(BASE_DIR) + "/" + str(self.output_dir) + "/logs", 
            f"MinP_GridSearch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        # 加载基础配置
        self.base_config = load_config(base_config_path, self.logger)
        validate_config(self.base_config)
        
        # 初始化vLLM模型（一次性创建，所有实验共享）
        self.model = None
        self._initialize_model()
        
        # 搜索结果
        self.search_results = []
    
    def _initialize_model(self):
        """
        初始化vLLM模型
        """
        try:
            from vllm import LLM
            
            model_config = self.base_config["model_config"]
            model_name = model_config["model_name"]
            model_path = model_config["model_path"] + model_name
            
            self.logger.info(f"正在初始化vLLM模型: {model_name}")
            self.model = LLM(
                model=model_path,
                tensor_parallel_size=model_config["tensor_parallel_size"],
                gpu_memory_utilization=model_config["gpu_memory_utilization"],
                max_model_len=model_config["max_model_len"],
                trust_remote_code=model_config["trust_remote_code"]
            )
            self.logger.info("vLLM模型初始化完成")
            
        except Exception as e:
            self.logger.error(f"vLLM模型初始化失败: {e}")
            raise
        
    def load_search_config(self, search_config_path: str) -> Dict[str, Any]:
        """
        加载搜索配置文件
        
        Args:
            search_config_path: 搜索配置文件路径
            
        Returns:
            Dict: 搜索配置字典
        """
        config_file = Path(search_config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"搜索配置文件未找到: {config_file}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                search_config = json.load(f)
            return search_config
        except json.JSONDecodeError as e:
            raise ValueError(f"搜索配置文件JSON格式错误: {e}")
        except Exception as e:
            raise RuntimeError(f"加载搜索配置文件失败: {e}")
    
    def validate_search_config(self, search_config: Dict[str, Any]) -> bool:
        """
        验证搜索配置文件的完整性
        
        Args:
            search_config: 搜索配置字典
            
        Returns:
            bool: 验证是否通过
        """
        required_sections = ["search_config", "search_space"]
        
        for section in required_sections:
            if section not in search_config:
                raise ValueError(f"搜索配置文件缺少必需的节: {section}")
        
        # 验证搜索空间
        search_space = search_config["search_space"]
        required_steps = ["step_1_values", "step_2_values", "step_3_values", "step_4_values"]
        
        for step in required_steps:
            if step not in search_space:
                raise ValueError(f"搜索空间缺少必需的步骤: {step}")
            
            values = search_space[step]
            if not isinstance(values, list) or len(values) == 0:
                raise ValueError(f"{step} 必须是非空列表")
            
            # 验证值的范围
            for i, val in enumerate(values):
                if not isinstance(val, (int, float)) or not 0.0 <= val <= 1.0:
                    raise ValueError(f"{step}[{i}] = {val} 不在有效范围[0.0, 1.0]内")
        
        # 验证温度值（如果存在）
        if "temperature_values" in search_space:
            temp_values = search_space["temperature_values"]
            if not isinstance(temp_values, list) or len(temp_values) == 0:
                raise ValueError("temperature_values 必须是非空列表")
            
            for i, val in enumerate(temp_values):
                if not isinstance(val, (int, float)) or not 0.0 < val <= 2.0:
                    raise ValueError(f"temperature_values[{i}] = {val} 不在有效范围(0.0, 2.0]内")
        
        return True
    
    def define_search_space_from_config(self, search_config: Dict[str, Any]) -> List[Tuple[List[float], float]]:
        """
        从配置文件定义搜索空间，包括min-p值和温度值
        
        Args:
            search_config: 搜索配置字典
            
        Returns:
            List[Tuple[List[float], float]]: 所有(min-p组合, 温度)的元组列表
        """
        search_space = search_config["search_space"]
        search_options = search_config.get("search_options", {})
        
        # 验证搜索空间配置
        required_steps = ["step_1_values", "step_2_values", "step_3_values", "step_4_values"]
        for step in required_steps:
            if step not in search_space:
                raise ValueError(f"搜索空间缺少必需的步骤: {step}")
            if not isinstance(search_space[step], list) or len(search_space[step]) == 0:
                raise ValueError(f"{step} 必须是非空列表")
        
        # 从配置文件的搜索空间生成笛卡尔积组合
        step_values = [
            search_space["step_1_values"],
            search_space["step_2_values"], 
            search_space["step_3_values"],
            search_space["step_4_values"]
        ]
        
        # 获取温度值（如果存在）
        temperature_values = search_space.get("temperature_values", [1.0])  # 默认温度为1.0
        
        # 生成所有可能的组合：(min-p值组合, 温度值)
        minp_combinations = list(itertools.product(*step_values))
        all_combinations = list(itertools.product(minp_combinations, temperature_values))
        
        # 转换为 (min-p列表, 温度) 的格式
        combinations = [(list(minp_combo), temp) for minp_combo, temp in all_combinations]
        
        self.logger.info(f"从配置文件搜索空间生成了 {len(combinations)} 个组合")
        self.logger.info(f"Step 1 值: {search_space['step_1_values']}")
        self.logger.info(f"Step 2 值: {search_space['step_2_values']}")
        self.logger.info(f"Step 3 值: {search_space['step_3_values']}")
        self.logger.info(f"Step 4 值: {search_space['step_4_values']}")
        self.logger.info(f"Temperature 值: {temperature_values}")
        
        # 检查最大实验数限制
        max_experiments = search_options.get("max_experiments", 1000)
        if len(combinations) > max_experiments:
            self.logger.warning(f"组合数量 ({len(combinations)}) 超过最大限制 ({max_experiments})，将截取前 {max_experiments} 个")
            combinations = combinations[:max_experiments]
        
        self.logger.info(f"最终搜索空间包含 {len(combinations)} 个组合")
        
        return combinations

    
    def create_config_for_combination(self, minp_values: List[float], temperature: float, experiment_id: int, run_id: int = 1) -> str:
        """
        为特定的min-p和温度组合创建配置文件
        
        Args:
            minp_values: min-p值列表
            temperature: 温度值
            experiment_id: 实验ID
            run_id: 运行次数ID
            
        Returns:
            str: 配置文件路径
        """
        # 深拷贝基础配置
        import copy
        config = copy.deepcopy(self.base_config)
        
        # 更新min-p值和温度值
        config["cot_config"]["step_minp_values"] = minp_values
        config["generation_config"]["temperature"] = temperature
        
        # 添加运行次数信息到配置中
        config["experiment_config"]["run_id"] = run_id
        
        min_p_str = '_'.join(map(str, minp_values))
        temp_str = str(temperature)
        
        # 为每组参数组合创建单独的文件夹，包含温度信息
        experiment_dir = Path(self.base_config["experiment_config"]["root_dir"]) / self.output_dir / f"minp_{min_p_str}_temp_{temp_str}"
        experiment_dir.mkdir(exist_ok=True, parents=True)
        
        # 更新输出目录 - 指向该实验的专用文件夹
        config["experiment_config"]["output_dir"] = f"{self.output_dir}/minp_{min_p_str}_temp_{temp_str}"
        
        # 创建配置文件 - 存放在该实验的文件夹中，包含run_id
        if run_id > 1:
            config_filename = f"config_experiment_{min_p_str}_temp_{temp_str}_run{run_id}.json"
        else:
            config_filename = f"config_experiment_{min_p_str}_temp_{temp_str}.json"
        config_path = experiment_dir / config_filename
        self.logger.info(f"创建配置文件: {config_path}")
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            
        return str(config_path)
    
    def run_single_experiment(self, minp_values: List[float], temperature: float, experiment_id: int, run_id: int = 1, num_runs: int = 1) -> Dict[str, Any]:
        """
        运行单个实验（使用共享的vLLM模型）
        
        Args:
            minp_values: min-p值列表
            temperature: 温度值
            experiment_id: 实验ID
            run_id: 当前运行次数 (1-based)
            num_runs: 总运行次数
            
        Returns:
            Dict: 实验结果
        """
        if num_runs > 1:
            self.logger.info(f"开始实验 {experiment_id} (第 {run_id}/{num_runs} 次): min-p值 = {minp_values}, 温度 = {temperature}")
        else:
            self.logger.info(f"开始实验 {experiment_id}: min-p值 = {minp_values}, 温度 = {temperature}")
        
        try:
            # 创建配置文件并运行实验（使用已创建的模型）
            config_path = self.create_config_for_combination(minp_values, temperature, experiment_id, run_id)
            
            # 运行实验，传入已创建的模型
            run_cot_experiment(config_path, model=self.model)
            
            # 读取结果
            result = self.extract_experiment_result(experiment_id, minp_values, temperature, run_id)
            
            # 清理临时配置文件
            try:
                Path(config_path).unlink()
            except Exception as e:
                self.logger.warning(f"删除临时配置文件失败: {e}")
            
            if num_runs > 1:
                self.logger.info(f"实验 {experiment_id} (第 {run_id}/{num_runs} 次) 完成，准确率: {result['accuracy']:.2f}%")
            else:
                self.logger.info(f"实验 {experiment_id} 完成，准确率: {result['accuracy']:.2f}%")
            
            return result
            
        except Exception as e:
            self.logger.error(f"实验 {experiment_id} (第 {run_id}/{num_runs} 次) 失败: {str(e)}")
            return {
                "experiment_id": experiment_id,
                "minp_values": minp_values,
                "temperature": temperature,
                "run_id": run_id,
                "accuracy": 0.0,
                "error": str(e),
                "status": "failed"
            }
    
    def extract_experiment_result(self, experiment_id: int, minp_values: List[float], temperature: float, run_id: int = 1) -> Dict[str, Any]:
        """
        从实验输出中提取结果
        
        Args:
            experiment_id: 实验ID
            minp_values: min-p值列表
            temperature: 温度值
            run_id: 运行次数ID
            
        Returns:
            Dict: 提取的结果
        """
        # 构建结果文件路径 - 现在结果文件保存在各自的参数组合文件夹中
        min_p_str = '_'.join(map(str, minp_values))
        temp_str = str(temperature)
        experiment_dir = Path(self.base_config["experiment_config"]["root_dir"]) / self.output_dir / f"minp_{min_p_str}_temp_{temp_str}"
        result_files = list(experiment_dir.glob("*.json"))
        # 排除配置文件，只获取结果文件
        result_files = [f for f in result_files if not f.name.startswith("config_")]
        
        if not result_files:
            raise FileNotFoundError(f"未找到min-p值为 {minp_values}, 温度为 {temperature} 的结果文件")
        
        # 读取最新的结果文件
        latest_result_file = max(result_files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_result_file, 'r', encoding='utf-8') as f:
            experiment_data = json.load(f)
        
        # 提取准确率信息 - 新格式中accuracy在results数组的最后一个元素
        summary = {}
        if experiment_data:
            # 检查新的文件格式（对象结构，包含results数组）
            if isinstance(experiment_data, dict) and "results" in experiment_data:
                results_array = experiment_data["results"]
                if results_array and isinstance(results_array, list):
                    summary = results_array[-1]  # 最后一个元素包含汇总信息
            # 兼容旧格式（直接是数组）
            elif isinstance(experiment_data, list):
                summary = experiment_data[-1] if experiment_data else {}
        
        accuracy = summary.get("accuracy", 0.0)
        correct_count = summary.get("correct_count", 0)
        total_samples = summary.get("total_samples", 0)
        
        return {
            "experiment_id": experiment_id,
            "minp_values": minp_values,
            "temperature": temperature,
            "run_id": run_id,
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_samples": total_samples,
            "result_file": str(latest_result_file),
            "status": "success"
        }
    
    def run_multiple_experiments(self, minp_values: List[float], temperature: float, experiment_id: int, num_runs: int) -> Dict[str, Any]:
        """
        运行多次实验并计算平均值
        
        Args:
            minp_values: min-p值列表
            temperature: 温度值
            experiment_id: 实验ID
            num_runs: 运行次数
            
        Returns:
            Dict: 平均实验结果
        """
        if num_runs <= 1:
            return self.run_single_experiment(minp_values, temperature, experiment_id, 1, 1)
        
        self.logger.info(f"开始运行实验 {experiment_id}，min-p值 = {minp_values}, 温度 = {temperature}，共 {num_runs} 次")
        
        all_runs = []
        successful_runs = []
        
        for run_id in range(1, num_runs + 1):
            try:
                result = self.run_single_experiment(minp_values, temperature, experiment_id, run_id, num_runs)
                all_runs.append(result)
                
                if result.get("status") == "success":
                    successful_runs.append(result)
                    
            except Exception as e:
                self.logger.error(f"实验 {experiment_id} 第 {run_id} 次运行失败: {str(e)}")
                failed_result = {
                    "experiment_id": experiment_id,
                    "minp_values": minp_values,
                    "temperature": temperature,
                    "run_id": run_id,
                    "accuracy": 0.0,
                    "error": str(e),
                    "status": "failed"
                }
                all_runs.append(failed_result)
        
        if not successful_runs:
            self.logger.error(f"实验 {experiment_id} 所有 {num_runs} 次运行都失败了")
            return {
                "experiment_id": experiment_id,
                "minp_values": minp_values,
                "temperature": temperature,
                "num_runs": num_runs,
                "successful_runs": 0,
                "accuracy": 0.0,
                "accuracy_std": 0.0,
                "accuracy_min": 0.0,
                "accuracy_max": 0.0,
                "correct_count": 0,
                "total_samples": 0,
                "all_runs": all_runs,
                "status": "failed"
            }
        
        # 计算统计信息
        accuracies = [run["accuracy"] for run in successful_runs]
        correct_counts = [run["correct_count"] for run in successful_runs]
        total_samples = successful_runs[0]["total_samples"]  # 假设所有运行的样本数相同
        
        avg_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies) if len(accuracies) > 1 else 0.0
        min_accuracy = np.min(accuracies)
        max_accuracy = np.max(accuracies)
        avg_correct_count = np.mean(correct_counts)
        
        self.logger.info(f"实验 {experiment_id} 完成: 平均准确率 {avg_accuracy:.2f}% ± {std_accuracy:.2f}% "
                        f"(范围: {min_accuracy:.2f}%-{max_accuracy:.2f}%), 成功运行 {len(successful_runs)}/{num_runs} 次")
        
        return {
            "experiment_id": experiment_id,
            "minp_values": minp_values,
            "temperature": temperature,
            "num_runs": num_runs,
            "successful_runs": len(successful_runs),
            "accuracy": avg_accuracy,
            "accuracy_std": std_accuracy,
            "accuracy_min": min_accuracy,
            "accuracy_max": max_accuracy,
            "correct_count": avg_correct_count,
            "total_samples": total_samples,
            "all_runs": all_runs,
            "status": "success" if successful_runs else "failed"
        }
    
    def run_grid_search_from_config(self, search_config_path: str) -> List[Dict[str, Any]]:
        """
        从配置文件运行网格搜索
        
        Args:
            search_config_path: 搜索配置文件路径
            
        Returns:
            List[Dict]: 所有实验结果
        """
        # 加载和验证搜索配置
        search_config = self.load_search_config(search_config_path)
        self.validate_search_config(search_config)
        
        
        # 定义搜索空间
        combinations = self.define_search_space_from_config(search_config)
        
        self.logger.info(f"开始基于配置文件的网格搜索，共 {len(combinations)} 个实验")
        
        # 获取实验设置
        experiment_settings = search_config.get("experiment_settings", {})
        continue_on_error = search_config.get("search_options", {}).get("continue_on_error", True)
        
        # 获取多次运行设置
        num_runs = experiment_settings.get("num_runs", 1)
        if num_runs > 1:
            self.logger.info(f"每个配置将运行 {num_runs} 次并取平均值")
        
        # 运行所有实验
        for i, (minp_values, temperature) in enumerate(combinations):
            try:
                # 使用多次运行功能
                result = self.run_multiple_experiments(minp_values, temperature, i + 1, num_runs)
                self.search_results.append(result)
                
                # 禁用中间结果保存以减少I/O开销
                # if search_config.get("search_options", {}).get("save_intermediate_results", True):
                #     self.save_intermediate_results()
                    
            except Exception as e:
                self.logger.error(f"实验 {i + 1} 执行失败: {str(e)}")
                if not continue_on_error:
                    raise
                else:
                    # 记录失败的实验
                    failed_result = {
                        "experiment_id": i + 1,
                        "minp_values": minp_values,
                        "temperature": temperature,
                        "num_runs": num_runs,
                        "successful_runs": 0,
                        "accuracy": 0.0,
                        "accuracy_std": 0.0,
                        "error": str(e),
                        "status": "failed"
                    }
                    self.search_results.append(failed_result)
        
        # 分析和保存最终结果
        self.analyze_and_save_results()
        
        return self.search_results

    def run_grid_search(self, 
                       min_values: List[float] = None,
                       max_values: List[float] = None,
                       num_points: int = 5) -> List[Dict[str, Any]]:
        """
        运行完整的网格搜索（保留原有功能以兼容性）
        
        Args:
            min_values: 每个步骤的最小min-p值
            max_values: 每个步骤的最大min-p值
            num_points: 每个维度的搜索点数
            
        Returns:
            List[Dict]: 所有实验结果
        """
        # 定义搜索空间
        combinations = self.define_search_space(min_values, max_values, num_points)
        
        self.logger.info(f"开始网格搜索，共 {len(combinations)} 个实验")
        
        # 运行所有实验
        for i, minp_values in enumerate(combinations):
            result = self.run_single_experiment(minp_values, i + 1)
            self.search_results.append(result)
            
        
        # 分析和保存最终结果
        self.analyze_and_save_results()
        
        return self.search_results
    

    
    def analyze_and_save_results(self):
        """分析并保存最终结果"""
        if not self.search_results:
            self.logger.warning("没有可分析的结果")
            return
        
        # 过滤成功的实验
        successful_results = [r for r in self.search_results if r.get("status") == "success"]
        
        if not successful_results:
            self.logger.error("没有成功的实验")
            return
        
        # 找到最佳和最差结果
        best_result = max(successful_results, key=lambda x: x["accuracy"])
        worst_result = min(successful_results, key=lambda x: x["accuracy"])
        
        # 计算统计信息
        accuracies = [r["accuracy"] for r in successful_results]
        
        # 为了方便分析，将数据转换为带step列的格式
        analysis_data = []
        for result in successful_results:
            # 处理多次运行的结果
            data_entry = {
                "experiment_id": result["experiment_id"],
                "minp_values": result["minp_values"],
                "temperature": result["temperature"],
                "step_1": result["minp_values"][0],
                "step_2": result["minp_values"][1],
                "step_3": result["minp_values"][2],
                "step_4": result["minp_values"][3],
                "accuracy": result["accuracy"],
                "correct_count": result["correct_count"],
                "total_samples": result["total_samples"]
            }
            
            # 如果是多次运行的结果，添加额外的统计信息
            if "num_runs" in result and result["num_runs"] > 1:
                data_entry.update({
                    "num_runs": result["num_runs"],
                    "successful_runs": result["successful_runs"],
                    "accuracy_std": result["accuracy_std"],
                    "accuracy_min": result["accuracy_min"],
                    "accuracy_max": result["accuracy_max"]
                })
            
            analysis_data.append(data_entry)
        
        # 按步骤和温度分析最佳值
        step_analysis = {}
        for step in [1, 2, 3, 4]:
            step_col = f"step_{step}"
            
            # 提取步骤值和准确率
            step_values = [data[step_col] for data in analysis_data]
            step_accuracies = [data["accuracy"] for data in analysis_data]
            
            # 计算相关性
            step_corr = np.corrcoef(step_values, step_accuracies)[0, 1] if len(set(step_values)) > 1 else 0.0
            
            # 找到最佳配置中的值
            step_best_value = best_result["minp_values"][step - 1]
            
            # 计算各值的平均准确率
            step_avg_by_value = {}
            for data in analysis_data:
                value = data[step_col]
                if value not in step_avg_by_value:
                    step_avg_by_value[value] = []
                step_avg_by_value[value].append(data["accuracy"])
            
            # 转换为平均值
            for value in step_avg_by_value:
                step_avg_by_value[value] = np.mean(step_avg_by_value[value])
            
            step_analysis[f"step_{step}"] = {
                "correlation_with_accuracy": step_corr,
                "best_value": step_best_value,
                "average_accuracy_by_value": step_avg_by_value
            }
        
        # 温度参数分析
        temp_values = [data["temperature"] for data in analysis_data]
        temp_accuracies = [data["accuracy"] for data in analysis_data]
        temp_corr = np.corrcoef(temp_values, temp_accuracies)[0, 1] if len(set(temp_values)) > 1 else 0.0
        temp_best_value = best_result["temperature"]
        
        temp_avg_by_value = {}
        for data in analysis_data:
            value = data["temperature"]
            if value not in temp_avg_by_value:
                temp_avg_by_value[value] = []
            temp_avg_by_value[value].append(data["accuracy"])
        
        for value in temp_avg_by_value:
            temp_avg_by_value[value] = np.mean(temp_avg_by_value[value])
        
        step_analysis["temperature"] = {
            "correlation_with_accuracy": temp_corr,
            "best_value": temp_best_value,
            "average_accuracy_by_value": temp_avg_by_value
        }
        
        stats = {
            "total_experiments": len(self.search_results),
            "successful_experiments": len(successful_results),
            "failed_experiments": len(self.search_results) - len(successful_results),
            "mean_accuracy": np.mean(accuracies),
            "std_accuracy": np.std(accuracies),
            "min_accuracy": np.min(accuracies),
            "max_accuracy": np.max(accuracies),
            "median_accuracy": np.median(accuracies)
        }
        
        # 获取准确率前十的结果
        top_10_results = sorted(successful_results, key=lambda x: x["accuracy"], reverse=True)[:10]
        
        # 保存完整结果
        final_results = {
            "summary_statistics": stats,
            "best_result": best_result,
            "worst_result": worst_result,
            "step_analysis": step_analysis,
            "top_10_results": top_10_results,
            "all_results": self.search_results,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        # 保存到文件
        results_file = Path(self.base_config["experiment_config"]["root_dir"]) / self.output_dir / f"grid_search_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        # 记录日志（按照analyze_best_minp.py的格式）
        self.logger.info("\n" + "="*80)
        self.logger.info("🎯 MIN-P 参数优化分析报告")
        self.logger.info("="*80)
        
        # 基础统计
        self.logger.info(f"\n📊 基础统计:")
        self.logger.info(f"   总实验数: {stats['total_experiments']}")
        self.logger.info(f"   成功实验数: {stats['successful_experiments']}")
        self.logger.info(f"   失败实验数: {stats['failed_experiments']}")
        self.logger.info(f"   平均准确率: {stats['mean_accuracy']:.2f}% ± {stats['std_accuracy']:.2f}%")
        self.logger.info(f"   准确率范围: {stats['min_accuracy']:.2f}% - {stats['max_accuracy']:.2f}%")
        self.logger.info(f"   中位数准确率: {stats['median_accuracy']:.2f}%")
        
        # 最佳结果
        self.logger.info(f"\n🏆 最佳配置:")
        self.logger.info(f"   Min-p值: {best_result['minp_values']}")
        self.logger.info(f"   温度: {best_result['temperature']}")
        self.logger.info(f"   准确率: {best_result['accuracy']:.2f}%")
        self.logger.info(f"   正确数/总数: {best_result['correct_count']}/{best_result['total_samples']}")
        
        # 最差结果
        self.logger.info(f"\n📉 最差配置:")
        self.logger.info(f"   Min-p值: {worst_result['minp_values']}")
        self.logger.info(f"   温度: {worst_result['temperature']}")
        self.logger.info(f"   准确率: {worst_result['accuracy']:.2f}%")
        
        # 步骤分析
        self.logger.info(f"\n🔍 各步骤参数分析:")
        for step, data in step_analysis.items():
            self.logger.info(f"\n   {step.upper().replace('_', ' ')}:")
            self.logger.info(f"     与准确率的相关性: {data['correlation_with_accuracy']:.3f}")
            self.logger.info(f"     最佳配置中的值: {data['best_value']}")
            self.logger.info(f"     各值的平均表现:")
            for value, avg_acc in sorted(data['average_accuracy_by_value'].items()):
                self.logger.info(f"       {value}: {avg_acc:.2f}%")
        
        # TOP结果
        self.logger.info(f"\n🌟 TOP-{len(top_10_results)} 配置:")
        for i, result in enumerate(top_10_results, 1):
            base_info = f"   {i:2d}. Min-p: {result['minp_values']}, Temp: {result['temperature']} -> {result['accuracy']:.2f}%"
            
            # 如果是多次运行的结果，显示额外信息
            if "num_runs" in result and result["num_runs"] > 1:
                std_info = f" ± {result['accuracy_std']:.2f}%"
                range_info = f" (范围: {result['accuracy_min']:.2f}%-{result['accuracy_max']:.2f}%)"
                runs_info = f" [{result['successful_runs']}/{result['num_runs']} 次成功]"
                self.logger.info(base_info + std_info + range_info + runs_info)
            else:
                self.logger.info(base_info)
        
        self.logger.info(f"\n💾 详细结果已保存到: {results_file}")
        self.logger.info("="*80)
        
        return final_results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Min-p 网格搜索工具")
    parser.add_argument(
        "--config", 
        type=str,
        default=str(BASE_DIR / "grid_search_config.json"),
        help="搜索配置文件路径 (JSON格式)"
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default=str(BASE_DIR / "config.json"),
        help="基础实验配置文件路径"
    )
    
    args = parser.parse_args()
    
    
    grid_search = MinPGridSearch(
        base_config_path=args.base_config,
        search_config_path=args.config
    )
    
    try:
        if args.config:
            # 使用配置文件进行搜索
            results = grid_search.run_grid_search_from_config(args.config)
        else:
            # 使用默认参数进行搜索
            print("未指定搜索配置文件，使用默认参数...")
            min_values = [0.0, 0.0, 0.0, 0.0]
            max_values = [0.3, 0.3, 0.3, 0.3]
            num_points = 3
            
            results = grid_search.run_grid_search(
                min_values=min_values,
                max_values=max_values,
                num_points=num_points
            )
        
        print("网格搜索完成！请查看日志和结果文件。")
        
    except Exception as e:
        print(f"搜索过程中出现错误: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
