#!/usr/bin/env python3
"""
运行统一Min-p值搜索的便捷脚本
"""

import os
import sys
from pathlib import Path

def main():
    """运行统一min-p值搜索"""
    
    print("=" * 60)
    print("统一Min-p值搜索")
    print("=" * 60)
    print("此脚本将测试所有阶段都使用相同的min-p值")
    print("搜索间隔: 0.1 (从0.0到1.0)")
    print("每个值将运行多次试验并取平均值")
    print("=" * 60)
    
    # 检查配置文件是否存在
    base_config = "config.json"
    search_config = "uniform_minp_search_config.json"
    
    if not Path(base_config).exists():
        print(f"错误: 基础配置文件 {base_config} 不存在")
        sys.exit(1)
    
    if not Path(search_config).exists():
        print(f"错误: 搜索配置文件 {search_config} 不存在")
        sys.exit(1)
    
    print(f"使用基础配置: {base_config}")
    print(f"使用搜索配置: {search_config}")
    print()
    
    # 询问用户是否继续
    response = input("是否开始搜索？ (y/n): ").strip().lower()
    if response not in ['y', 'yes', '是']:
        print("搜索已取消")
        sys.exit(0)
    
    # 运行搜索
    try:
        from uniform_minp_search import UniformMinPSearch
        
        print("\n开始初始化搜索器...")
        searcher = UniformMinPSearch(base_config, search_config)
        
        print("开始运行搜索...")
        results = searcher.run_search()
        
        print(f"\n搜索完成！")
        print(f"结果保存在: {searcher.output_dir}")
        print(f"总共测试了 {len(results)} 个配置")
        
        # 显示最佳结果
        successful_results = [r for r in results if r.get("status") == "completed" and r.get("mean_accuracy", 0) > 0]
        if successful_results:
            best_result = max(successful_results, key=lambda x: x["mean_accuracy"])
            print(f"\n最佳配置:")
            print(f"  统一min-p值: {best_result['uniform_minp']}")
            print(f"  温度: {best_result['temperature']}")
            print(f"  平均准确率: {best_result['mean_accuracy']:.2f}%")
            if 'std_accuracy' in best_result:
                print(f"  标准差: ±{best_result['std_accuracy']:.2f}%")
        
    except KeyboardInterrupt:
        print("\n\n搜索被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()