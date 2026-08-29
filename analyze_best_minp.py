#!/usr/bin/env python3
"""
Min-p 结果分析脚本
从已生成的实验结果中寻找最优的 min-p 值组合
"""

import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime
import argparse

class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理NumPy数据类型"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class MinPResultAnalyzer:
    """Min-p 结果分析器"""
    
    def __init__(self, results_dir: str = "grid_search_results"):
        """
        初始化分析器
        
        Args:
            results_dir: 结果目录路径
        """
        self.results_dir = Path(results_dir)
        self.base_dir = Path(__file__).parent
        self.results = []
        
    def scan_experiment_directories(self) -> List[Dict[str, Any]]:
        """
        扫描所有实验目录，提取结果
        
        Returns:
            List: 所有实验结果列表
        """
        all_results = []
        
        # 查找所有 minp_* 目录
        minp_dirs = list(self.results_dir.glob("minp_*"))
        
        print(f"找到 {len(minp_dirs)} 个实验目录")
        
        for minp_dir in minp_dirs:
            try:
                # 从目录名提取 min-p 值和temperature
                dir_name = minp_dir.name
                # 解析格式: minp_0.6_0.4_0.8_1_temp_1.5
                parts = dir_name.split("_temp_")
                if len(parts) == 2:
                    minp_part = parts[0].replace("minp_", "")
                    temp_part = parts[1]
                    minp_values = [float(x) for x in minp_part.split("_")]
                    temperature = float(temp_part)
                else:
                    # 兼容旧格式（没有temperature）
                    minp_values_str = dir_name.replace("minp_", "")
                    minp_values = [float(x) for x in minp_values_str.split("_")]
                    temperature = None
                
                # 查找所有结果文件（排除配置文件）
                result_files = [f for f in minp_dir.glob("*.json") 
                              if not f.name.startswith("config_")]
                
                if not result_files:
                    print(f"警告: {minp_dir} 中未找到结果文件")
                    continue
                
                # 读取所有结果文件而不是只读取最新的
                experiment_results = []
                for result_file in result_files:
                    try:
                        with open(result_file, 'r', encoding='utf-8') as f:
                            experiment_data = json.load(f)
                        
                        # 提取汇总信息 - 新格式中accuracy在results数组的最后一个元素
                        if experiment_data:
                            summary = None
                            # 检查新的文件格式（对象结构，包含results数组）
                            if isinstance(experiment_data, dict) and "results" in experiment_data:
                                results_array = experiment_data["results"]
                                if results_array and isinstance(results_array, list):
                                    summary = results_array[-1]  # 最后一个元素包含汇总信息
                            # 兼容旧格式（直接是数组）
                            elif isinstance(experiment_data, list):
                                summary = experiment_data[-1]
                            
                            if summary and "accuracy" in summary:
                                result = {
                                    "minp_values": minp_values,
                                    "step_1": minp_values[0],
                                    "step_2": minp_values[1], 
                                    "step_3": minp_values[2],
                                    "step_4": minp_values[3],
                                    "temperature": temperature,
                                    "accuracy": summary.get("accuracy", 0.0),
                                    "correct_count": summary.get("correct_count", 0),
                                    "total_samples": summary.get("total_samples", 0),
                                    "result_file": str(result_file),
                                    "experiment_dir": str(minp_dir)
                                }
                                experiment_results.append(result)
                    except Exception as e:
                        print(f"警告: 读取结果文件 {result_file} 时出错: {e}")
                        continue
                
                if experiment_results:
                    all_results.extend(experiment_results)
                    temp_str = f", temperature: {temperature}" if temperature is not None else ""
                    print(f"✓ 成功解析: {dir_name} - 找到 {len(experiment_results)} 次实验{temp_str}")
                else:
                    print(f"警告: {minp_dir} 中没有有效的结果文件")
                    
            except Exception as e:
                print(f"错误: 解析 {minp_dir} 时出错: {e}")
                continue
        
        # 聚合相同配置的实验结果
        aggregated_results = self.aggregate_experiment_results(all_results)
        return aggregated_results
    
    def aggregate_experiment_results(self, all_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        聚合相同配置的多次实验结果，计算平均值和标准差
        
        Args:
            all_results: 所有实验结果列表
            
        Returns:
            List: 聚合后的结果列表
        """
        from collections import defaultdict
        
        # 按配置分组
        grouped_results = defaultdict(list)
        
        for result in all_results:
            # 创建配置键：包含min-p值和temperature
            config_key = tuple(result["minp_values"])
            if result["temperature"] is not None:
                config_key = config_key + (result["temperature"],)
            
            grouped_results[config_key].append(result)
        
        # 计算每组的平均值和标准差
        aggregated_results = []
        for config_key, group_results in grouped_results.items():
            if len(config_key) == 5:  # 包含temperature
                minp_values = list(config_key[:4])
                temperature = config_key[4]
            else:  # 不包含temperature
                minp_values = list(config_key)
                temperature = None
            
            # 提取所有准确率
            accuracies = [r["accuracy"] for r in group_results]
            correct_counts = [r["correct_count"] for r in group_results]
            total_samples = [r["total_samples"] for r in group_results]
            
            # 计算统计信息
            mean_accuracy = float(np.mean(accuracies))
            std_accuracy = float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else 0.0
            mean_correct = float(np.mean(correct_counts))
            mean_total = float(np.mean(total_samples))
            
            # 创建聚合结果
            aggregated_result = {
                "minp_values": minp_values,
                "step_1": minp_values[0],
                "step_2": minp_values[1], 
                "step_3": minp_values[2],
                "step_4": minp_values[3],
                "temperature": temperature,
                "accuracy": mean_accuracy,
                "accuracy_std": std_accuracy,
                "correct_count": mean_correct,
                "total_samples": mean_total,
                "num_runs": len(group_results),
                "all_accuracies": accuracies,  # 保留所有原始准确率用于分析
                "experiment_dirs": [r["experiment_dir"] for r in group_results]
            }
            
            aggregated_results.append(aggregated_result)
            
            # 打印聚合信息
            temp_str = f", T: {temperature}" if temperature is not None else ""
            if len(group_results) > 1:
                print(f"📊 聚合配置 {minp_values}{temp_str}: {len(group_results)} 次实验, "
                      f"平均准确率: {mean_accuracy:.2f}% ± {std_accuracy:.2f}%")
            else:
                print(f"📊 单次配置 {minp_values}{temp_str}: 准确率: {mean_accuracy:.2f}%")
        
        print(f"\n总共聚合了 {len(aggregated_results)} 个不同的配置")
        return aggregated_results
    
    def load_grid_search_results(self) -> List[Dict[str, Any]]:
        """
        加载网格搜索的汇总结果文件
        
        Returns:
            List: 实验结果列表
        """
        # 查找网格搜索结果文件（支持多种命名格式）
        grid_result_files = list(self.results_dir.glob("grid_search_results*.json"))
        
        if not grid_result_files:
            print("未找到网格搜索汇总结果文件，尝试从单个实验目录提取...")
            return self.scan_experiment_directories()
        
        # 使用最新的结果文件
        latest_grid_file = max(grid_result_files, key=lambda x: x.stat().st_mtime)
        print(f"加载网格搜索结果文件: {latest_grid_file}")
        
        try:
            with open(latest_grid_file, 'r', encoding='utf-8') as f:
                grid_data = json.load(f)
            
            # 处理新格式的汇总文件
            if "all_results" in grid_data:
                # 过滤成功的实验并转换格式
                all_results = []
                successful_results = [r for r in grid_data["all_results"] 
                                    if r.get("status") == "success"]
                
                for result in successful_results:
                    minp_values = result.get("minp_values", [])
                    if len(minp_values) >= 4:
                        processed_result = {
                            "minp_values": minp_values,
                            "step_1": minp_values[0],
                            "step_2": minp_values[1], 
                            "step_3": minp_values[2],
                            "step_4": minp_values[3],
                            "temperature": result.get("temperature"),
                            "accuracy": result.get("accuracy", 0.0),
                            "correct_count": result.get("correct_count", 0),
                            "total_samples": result.get("total_samples", 0),
                            "experiment_id": result.get("experiment_id")
                        }
                        all_results.append(processed_result)
                
                if all_results:
                    print(f"从汇总文件中找到 {len(all_results)} 个成功的实验")
                    # 对汇总文件中的结果也进行聚合处理
                    aggregated_results = self.aggregate_experiment_results(all_results)
                    return aggregated_results
                else:
                    print("汇总文件中没有有效的实验，尝试从单个目录提取...")
                    return self.scan_experiment_directories()
            else:
                print("汇总文件格式不正确，尝试从单个目录提取...")
                return self.scan_experiment_directories()
                
        except Exception as e:
            print(f"读取汇总文件失败: {e}")
            print("尝试从单个实验目录提取...")
            return self.scan_experiment_directories()
    
    def analyze_results(self) -> Dict[str, Any]:
        """
        分析所有结果，找出最优配置
        
        Returns:
            Dict: 分析结果
        """
        # 加载结果
        self.results = self.load_grid_search_results()
        
        if not self.results:
            raise ValueError("未找到任何有效的实验结果")
        
        print(f"\n总共分析 {len(self.results)} 个实验结果")
        
        # 转换为DataFrame便于分析
        df = pd.DataFrame(self.results)
        
        # 基础统计
        stats = {
            "total_experiments": len(df),
            "total_runs": df["num_runs"].sum() if "num_runs" in df.columns else len(df),
            "mean_accuracy": df["accuracy"].mean(),
            "std_accuracy": df["accuracy"].std(),
            "min_accuracy": df["accuracy"].min(),
            "max_accuracy": df["accuracy"].max(),
            "median_accuracy": df["accuracy"].median()
        }
        
        # 添加聚合统计信息
        if "accuracy_std" in df.columns:
            stats["mean_accuracy_std"] = df["accuracy_std"].mean()
            stats["experiments_with_multiple_runs"] = (df["num_runs"] > 1).sum() if "num_runs" in df.columns else 0
        
        # 找出最佳结果
        best_idx = df["accuracy"].idxmax()
        best_result = df.loc[best_idx].to_dict()
        
        # 找出最差结果
        worst_idx = df["accuracy"].idxmin()
        worst_result = df.loc[worst_idx].to_dict()
        
        # 按步骤分析最佳值
        step_analysis = {}
        for step in [1, 2, 3, 4]:
            step_col = f"step_{step}"
            if step_col in df.columns:
                step_corr = df[step_col].corr(df["accuracy"])
                step_best_value = df.loc[df["accuracy"].idxmax(), step_col]
                step_avg_by_value = df.groupby(step_col)["accuracy"].mean().to_dict()
                
                step_analysis[f"step_{step}"] = {
                    "correlation_with_accuracy": step_corr,
                    "best_value": step_best_value,
                    "average_accuracy_by_value": step_avg_by_value
                }
        
        # Temperature分析
        temperature_analysis = {}
        if "temperature" in df.columns and df["temperature"].notna().any():
            temp_corr = df["temperature"].corr(df["accuracy"])
            temp_best_value = df.loc[df["accuracy"].idxmax(), "temperature"]
            temp_avg_by_value = df.groupby("temperature")["accuracy"].mean().to_dict()
            
            temperature_analysis = {
                "correlation_with_accuracy": temp_corr,
                "best_value": temp_best_value,
                "average_accuracy_by_value": temp_avg_by_value
            }
        
        # 获取TOP-K结果
        top_k = min(10, len(df))
        top_columns = ["minp_values", "accuracy", "correct_count", "total_samples"]
        if "temperature" in df.columns:
            top_columns.append("temperature")
        if "accuracy_std" in df.columns:
            top_columns.append("accuracy_std")
        if "num_runs" in df.columns:
            top_columns.append("num_runs")
        
        top_results = df.nlargest(top_k, "accuracy")[top_columns].to_dict("records")
        
        analysis_result = {
            "summary_statistics": stats,
            "best_result": best_result,
            "worst_result": worst_result,
            "step_analysis": step_analysis,
            "temperature_analysis": temperature_analysis,
            f"top_{top_k}_results": top_results,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return analysis_result
    
    def print_analysis_report(self, analysis: Dict[str, Any]):
        """
        打印分析报告
        
        Args:
            analysis: 分析结果
        """
        print("\n" + "="*80)
        print("🎯 MIN-P 参数优化分析报告")
        print("="*80)
        
        # 基础统计
        stats = analysis["summary_statistics"]
        print(f"\n📊 基础统计:")
        print(f"   总配置数: {stats['total_experiments']}")
        if "total_runs" in stats and stats["total_runs"] != stats["total_experiments"]:
            print(f"   总运行次数: {stats['total_runs']}")
        if "experiments_with_multiple_runs" in stats and stats["experiments_with_multiple_runs"] > 0:
            print(f"   多次运行的配置: {stats['experiments_with_multiple_runs']}")
        print(f"   平均准确率: {stats['mean_accuracy']:.2f}% ± {stats['std_accuracy']:.2f}%")
        if "mean_accuracy_std" in stats:
            print(f"   配置内平均标准差: {stats['mean_accuracy_std']:.3f}%")
        print(f"   准确率范围: {stats['min_accuracy']:.2f}% - {stats['max_accuracy']:.2f}%")
        print(f"   中位数准确率: {stats['median_accuracy']:.2f}%")
        
        # 最佳结果
        best = analysis["best_result"]
        print(f"\n🏆 最佳配置:")
        print(f"   Min-p值: {best['minp_values']}")
        if 'temperature' in best and best['temperature'] is not None:
            print(f"   Temperature: {best['temperature']}")
        print(f"   准确率: {best['accuracy']:.2f}%")
        print(f"   正确数/总数: {best['correct_count']}/{best['total_samples']}")
        if 'accuracy_std' in best and best['accuracy_std'] is not None:
            print(f"   准确率标准差: {best['accuracy_std']:.3f}%")
        if 'num_runs' in best and best['num_runs'] is not None:
            print(f"   运行次数: {best['num_runs']}")
        
        # 最差结果
        worst = analysis["worst_result"]
        print(f"\n📉 最差配置:")
        print(f"   Min-p值: {worst['minp_values']}")
        if 'temperature' in worst and worst['temperature'] is not None:
            print(f"   Temperature: {worst['temperature']}")
        print(f"   准确率: {worst['accuracy']:.2f}%")
        
        # 步骤分析
        print(f"\n🔍 各步骤参数分析:")
        step_analysis = analysis["step_analysis"]
        for step, data in step_analysis.items():
            print(f"\n   {step.upper().replace('_', ' ')}:")
            print(f"     与准确率的相关性: {data['correlation_with_accuracy']:.3f}")
            print(f"     最佳配置中的值: {data['best_value']}")
            print(f"     各值的平均表现:")
            for value, avg_acc in sorted(data['average_accuracy_by_value'].items()):
                print(f"       {value}: {avg_acc:.2f}%")
        
        # Temperature分析
        temperature_analysis = analysis.get("temperature_analysis", {})
        if temperature_analysis:
            print(f"\n🌡️ Temperature参数分析:")
            print(f"   与准确率的相关性: {temperature_analysis['correlation_with_accuracy']:.3f}")
            print(f"   最佳配置中的值: {temperature_analysis['best_value']}")
            print(f"   各值的平均表现:")
            for value, avg_acc in sorted(temperature_analysis['average_accuracy_by_value'].items()):
                print(f"     {value}: {avg_acc:.2f}%")
        
        # TOP结果
        # 找到正确的top结果键名
        top_key = None
        for key in analysis.keys():
            if key.startswith("top_") and key.endswith("_results"):
                top_key = key
                break
        
        if top_key:
            top_results = analysis[top_key]
            print(f"\n🌟 TOP-{len(top_results)} 配置:")
            for i, result in enumerate(top_results, 1):
                temp_str = f", T: {result['temperature']}" if 'temperature' in result and result['temperature'] is not None else ""
                
                # 构建显示字符串
                display_parts = [f"Min-p: {result['minp_values']}{temp_str}"]
                display_parts.append(f"准确率: {result['accuracy']:.2f}%")
                
                # 添加标准差信息（如果有）
                if 'accuracy_std' in result and result['accuracy_std'] is not None and result['accuracy_std'] > 0:
                    display_parts.append(f"±{result['accuracy_std']:.2f}%")
                
                # 添加运行次数信息（如果有多次运行）
                if 'num_runs' in result and result['num_runs'] is not None and result['num_runs'] > 1:
                    display_parts.append(f"({result['num_runs']}次)")
                
                print(f"   {i:2d}. {' '.join(display_parts)}")
        
        print("\n" + "="*80)
    
    def save_analysis_results(self, analysis: Dict[str, Any], output_file: str = None):
        """
        保存分析结果到文件
        
        Args:
            analysis: 分析结果
            output_file: 输出文件路径
        """
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.base_dir / f"minp_analysis_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        print(f"\n💾 分析结果已保存到: {output_file}")
    
    def export_to_csv(self, output_file: str = None):
        """
        导出结果到CSV文件
        
        Args:
            output_file: 输出文件路径
        """
        if not self.results:
            print("没有结果可导出")
            return
        
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.base_dir / f"minp_results_{timestamp}.csv"
        
        df = pd.DataFrame(self.results)
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"📄 结果已导出到CSV: {output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Min-p 结果分析工具")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="/cephfs/shared/sunyifan/Min-p-CoT/grid_search_results/Llama-3.2-3B-Instruct_GSM8k_test_0.2_interval",
        help="结果目录路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="分析结果输出文件路径"
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="导出结果到CSV文件"
    )
    
    args = parser.parse_args()
    
    try:
        # 创建分析器
        analyzer = MinPResultAnalyzer(args.results_dir)
        
        # 执行分析
        print("🔍 开始分析Min-p实验结果...")
        analysis = analyzer.analyze_results()
        
        # 打印报告
        analyzer.print_analysis_report(analysis)
        
        # 保存结果
        analyzer.save_analysis_results(analysis, args.output)
        
        # 导出CSV（如果需要）
        if args.export_csv:
            analyzer.export_to_csv()
        
        print("\n✅ 分析完成！")
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())