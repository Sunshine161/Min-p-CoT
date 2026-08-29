#!/usr/bin/env python3
"""
统一Min-p值搜索脚本
用于寻找所有阶段都使用相同min-p值的最佳配置
"""

import json
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Any
import numpy as np
import statistics

from min_p_cot import run_cot_experiment, load_config, validate_config
from logger import setup_logger

BASE_DIR = Path(__file__).parent

class UniformMinPSearch:
    """统一Min-p值搜索类"""
    
    def __init__(self, base_config_path: str, search_config_path: str):
        """
        初始化统一min-p值搜索
        
        Args:
            base_config_path: 基础配置文件路径
            search_config_path: 搜索配置文件路径
        """
        self.base_config_path = base_config_path
        self.search_config = self.load_search_config(search_config_path)
        
        # 创建输出目录
        model_name = load_config(base_config_path)["model_config"]["model_name"]
        dataset_name = load_config(base_config_path)["dataset_config"]["dataset_name"]
        search_interval = self.search_config["search_config"]["search_interval"]
        self.output_dir = Path(self.search_config["search_config"]["output_dir"] + model_name + "_" + dataset_name + "_uniform_" + search_interval)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # 设置日志
        self.logger = setup_logger(
            str(BASE_DIR) + "/" + str(self.output_dir) + "/logs", 
            f"UniformMinP_Search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    
    def get_uniform_minp_combinations(self) -> List[Tuple[float, float]]:
        """
        获取统一min-p值和温度的组合
        
        Returns:
            List[Tuple[float, float]]: (uniform_minp_value, temperature)的元组列表
        """
        search_space = self.search_config["search_space"]
        
        # 获取统一min-p值
        uniform_minp_values = search_space.get("uniform_minp_values", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        
        # 获取温度值
        temperature_values = search_space.get("temperature_values", [1.0])
        
        # 生成所有组合
        combinations = []
        for minp_val in uniform_minp_values:
            for temp_val in temperature_values:
                combinations.append((minp_val, temp_val))
        
        self.logger.info(f"生成了 {len(combinations)} 个统一min-p值组合")
        self.logger.info(f"统一Min-p值: {uniform_minp_values}")
        self.logger.info(f"温度值: {temperature_values}")
        
        return combinations
    
    def create_config_for_uniform_minp(self, uniform_minp: float, temperature: float, experiment_id: int, run_id: int = 1) -> str:
        """
        为统一min-p值创建配置文件
        
        Args:
            uniform_minp: 统一的min-p值（所有步骤都使用此值）
            temperature: 温度值
            experiment_id: 实验ID
            run_id: 运行次数ID
            
        Returns:
            str: 配置文件路径
        """
        # 深拷贝基础配置
        import copy
        config = copy.deepcopy(self.base_config)
        
        # 设置所有步骤都使用相同的min-p值
        config["cot_config"]["step_minp_values"] = [uniform_minp, uniform_minp, uniform_minp, uniform_minp]
        config["generation_config"]["temperature"] = temperature
        
        # 添加运行次数信息到配置中
        config["experiment_config"]["run_id"] = run_id
        
        # 为每组参数组合创建单独的文件夹
        experiment_dir = Path(self.base_config["experiment_config"]["root_dir"]) / self.output_dir / f"uniform_minp_{uniform_minp}_temp_{temperature}"
        experiment_dir.mkdir(exist_ok=True, parents=True)
        
        # 更新输出目录
        config["experiment_config"]["output_dir"] = f"{self.output_dir}/uniform_minp_{uniform_minp}_temp_{temperature}"
        
        # 创建配置文件
        if run_id > 1:
            config_filename = f"config_uniform_minp_{uniform_minp}_temp_{temperature}_run{run_id}.json"
        else:
            config_filename = f"config_uniform_minp_{uniform_minp}_temp_{temperature}.json"
        config_path = experiment_dir / config_filename
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            
        return str(config_path)
    
    def run_single_experiment(self, uniform_minp: float, temperature: float, experiment_id: int, run_id: int = 1, num_runs: int = 1) -> Dict[str, Any]:
        """
        运行单个实验
        
        Args:
            uniform_minp: 统一的min-p值
            temperature: 温度值
            experiment_id: 实验ID
            run_id: 当前运行次数
            num_runs: 总运行次数
            
        Returns:
            Dict: 实验结果
        """
        if num_runs > 1:
            self.logger.info(f"开始实验 {experiment_id} (第 {run_id}/{num_runs} 次): 统一min-p值 = {uniform_minp}, 温度 = {temperature}")
        else:
            self.logger.info(f"开始实验 {experiment_id}: 统一min-p值 = {uniform_minp}, 温度 = {temperature}")
        
        try:
            # 创建配置文件并运行实验
            config_path = self.create_config_for_uniform_minp(uniform_minp, temperature, experiment_id, run_id)
            
            # 运行实验，传入已创建的模型
            run_cot_experiment(config_path, model=self.model)
            
            # 读取结果
            result = self.extract_experiment_result(experiment_id, uniform_minp, temperature, run_id)
            
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
                "uniform_minp": uniform_minp,
                "temperature": temperature,
                "run_id": run_id,
                "accuracy": 0.0,
                "error": str(e),
                "status": "failed"
            }
    
    def extract_experiment_result(self, experiment_id: int, uniform_minp: float, temperature: float, run_id: int = 1) -> Dict[str, Any]:
        """
        从实验输出中提取结果
        """
        # 构建结果文件路径
        experiment_dir = Path(self.base_config["experiment_config"]["root_dir"]) / self.output_dir / f"uniform_minp_{uniform_minp}_temp_{temperature}"
        result_files = list(experiment_dir.glob("*.json"))
        # 排除配置文件，只获取结果文件
        result_files = [f for f in result_files if not f.name.startswith("config_")]
        
        if not result_files:
            raise FileNotFoundError(f"未找到统一min-p值为 {uniform_minp}, 温度为 {temperature} 的结果文件")
        
        # 读取最新的结果文件
        latest_result_file = max(result_files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_result_file, 'r', encoding='utf-8') as f:
            experiment_data = json.load(f)
        
        # 提取准确率信息
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
        total_count = summary.get("total_count", 0)
        
        return {
            "experiment_id": experiment_id,
            "uniform_minp": uniform_minp,
            "temperature": temperature,
            "run_id": run_id,
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total_count,
            "status": "completed"
        }
    
    def run_multiple_experiments(self, uniform_minp: float, temperature: float, experiment_id: int, num_runs: int) -> Dict[str, Any]:
        """
        运行多次实验并计算平均值
        
        Args:
            uniform_minp: 统一的min-p值
            temperature: 温度值
            experiment_id: 实验ID
            num_runs: 运行次数
            
        Returns:
            Dict: 包含平均结果的字典
        """
        self.logger.info(f"开始运行 {num_runs} 次实验: 统一min-p值 = {uniform_minp}, 温度 = {temperature}")
        
        results = []
        for run_id in range(1, num_runs + 1):
            result = self.run_single_experiment(uniform_minp, temperature, experiment_id, run_id, num_runs)
            results.append(result)
        
        # 计算平均值和统计信息
        accuracies = [r["accuracy"] for r in results if r["status"] == "completed"]
        
        if not accuracies:
            self.logger.error(f"所有 {num_runs} 次实验都失败了")
            return {
                "experiment_id": experiment_id,
                "uniform_minp": uniform_minp,
                "temperature": temperature,
                "num_runs": num_runs,
                "mean_accuracy": 0.0,
                "std_accuracy": 0.0,
                "min_accuracy": 0.0,
                "max_accuracy": 0.0,
                "successful_runs": 0,
                "failed_runs": num_runs,
                "status": "all_failed",
                "individual_results": results
            }
        
        mean_accuracy = statistics.mean(accuracies)
        std_accuracy = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
        min_accuracy = min(accuracies)
        max_accuracy = max(accuracies)
        successful_runs = len(accuracies)
        failed_runs = num_runs - successful_runs
        
        self.logger.info(f"实验 {experiment_id} 完成: 统一min-p值 = {uniform_minp}, 温度 = {temperature}")
        self.logger.info(f"平均准确率: {mean_accuracy:.2f}% (±{std_accuracy:.2f}%)")
        self.logger.info(f"成功运行: {successful_runs}/{num_runs}")
        
        return {
            "experiment_id": experiment_id,
            "uniform_minp": uniform_minp,
            "temperature": temperature,
            "num_runs": num_runs,
            "mean_accuracy": mean_accuracy,
            "std_accuracy": std_accuracy,
            "min_accuracy": min_accuracy,
            "max_accuracy": max_accuracy,
            "successful_runs": successful_runs,
            "failed_runs": failed_runs,
            "status": "completed",
            "individual_results": results
        }
    
    def run_search(self) -> List[Dict[str, Any]]:
        """
        运行完整的统一min-p值搜索
        
        Returns:
            List[Dict]: 所有实验结果
        """
        self.logger.info("开始统一Min-p值搜索")
        
        # 获取搜索组合
        combinations = self.get_uniform_minp_combinations()
        
        # 获取实验设置
        experiment_settings = self.search_config.get("experiment_settings", {})
        num_runs = experiment_settings.get("num_runs", 3)
        
        self.logger.info(f"将测试 {len(combinations)} 个组合，每个组合运行 {num_runs} 次")
        
        all_results = []
        
        for i, (uniform_minp, temperature) in enumerate(combinations, 1):
            try:
                result = self.run_multiple_experiments(uniform_minp, temperature, i, num_runs)
                all_results.append(result)
                self.search_results.append(result)
                
                # 保存中间结果
                if self.search_config.get("search_options", {}).get("save_intermediate_results", True):
                    self.save_intermediate_results(all_results)
                    
            except Exception as e:
                self.logger.error(f"实验组合 {i} 失败: {str(e)}")
                failed_result = {
                    "experiment_id": i,
                    "uniform_minp": uniform_minp,
                    "temperature": temperature,
                    "num_runs": num_runs,
                    "mean_accuracy": 0.0,
                    "status": "failed",
                    "error": str(e)
                }
                all_results.append(failed_result)
                self.search_results.append(failed_result)
        
        # 保存最终结果
        self.save_final_results(all_results)
        
        # 分析结果
        self.analyze_results(all_results)
        
        self.logger.info("统一Min-p值搜索完成")
        return all_results
    
    def save_intermediate_results(self, results: List[Dict[str, Any]]):
        """保存中间结果"""
        results_file = self.output_dir / "intermediate_results.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    def save_final_results(self, results: List[Dict[str, Any]]):
        """保存最终结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.output_dir / f"uniform_minp_search_results_{timestamp}.json"
        
        # 准备完整的结果数据
        full_results = {
            "search_config": self.search_config,
            "base_config": self.base_config,
            "timestamp": timestamp,
            "total_experiments": len(results),
            "results": results
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"最终结果已保存到: {results_file}")
    
    def analyze_results(self, results: List[Dict[str, Any]]):
        """分析搜索结果"""
        self.logger.info("=" * 60)
        self.logger.info("统一Min-p值搜索结果分析")
        self.logger.info("=" * 60)
        
        # 筛选成功的实验
        successful_results = [r for r in results if r.get("status") == "completed" and r.get("mean_accuracy", 0) > 0]
        
        if not successful_results:
            self.logger.warning("没有成功的实验结果")
            return
        
        # 按平均准确率排序
        successful_results.sort(key=lambda x: x["mean_accuracy"], reverse=True)
        
        self.logger.info(f"成功完成的实验: {len(successful_results)}/{len(results)}")
        self.logger.info("")
        
        # 显示前10个最佳结果
        self.logger.info("前10个最佳统一Min-p值配置:")
        for i, result in enumerate(successful_results[:10], 1):
            self.logger.info(f"第{i}名: 统一min-p={result['uniform_minp']}, 温度={result['temperature']}, "
                           f"平均准确率={result['mean_accuracy']:.2f}% (±{result.get('std_accuracy', 0):.2f}%)")
        
        # 保存最佳配置
        if self.search_config.get("result_analysis", {}).get("export_best_config", True):
            self.export_best_config(successful_results[0])
        
        self.logger.info("=" * 60)
    
    def export_best_config(self, best_result: Dict[str, Any]):
        """导出最佳配置"""
        # 创建最佳配置
        import copy
        best_config = copy.deepcopy(self.base_config)
        
        uniform_minp = best_result["uniform_minp"]
        temperature = best_result["temperature"]
        
        # 设置最佳参数
        best_config["cot_config"]["step_minp_values"] = [uniform_minp, uniform_minp, uniform_minp, uniform_minp]
        best_config["generation_config"]["temperature"] = temperature
        
        # 添加搜索结果信息
        best_config["search_result"] = {
            "mean_accuracy": best_result["mean_accuracy"],
            "std_accuracy": best_result.get("std_accuracy", 0),
            "uniform_minp": uniform_minp,
            "temperature": temperature,
            "search_timestamp": datetime.now().isoformat()
        }
        
        # 保存最佳配置
        best_config_file = self.output_dir / "best_uniform_minp_config.json"
        with open(best_config_file, 'w', encoding='utf-8') as f:
            json.dump(best_config, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"最佳统一min-p配置已导出到: {best_config_file}")
        self.logger.info(f"最佳统一min-p值: {uniform_minp}, 温度: {temperature}, 平均准确率: {best_result['mean_accuracy']:.2f}%")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="统一Min-p值搜索")
    parser.add_argument("--base-config", default="/cephfs/shared/sunyifan/Min-p-CoT/config.json", help="基础配置文件路径")
    parser.add_argument("--search-config", default="/cephfs/shared/sunyifan/Min-p-CoT/uniform_minp_search_config.json", help="搜索配置文件路径")
    
    args = parser.parse_args()
    
    try:
        # 创建搜索器并运行
        searcher = UniformMinPSearch(args.base_config, args.search_config)
        results = searcher.run_search()
        
        print(f"搜索完成！结果已保存到: {searcher.output_dir}")
        
    except Exception as e:
        print(f"搜索失败: {str(e)}")
        raise


if __name__ == "__main__":
    main()