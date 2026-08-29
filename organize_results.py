#!/usr/bin/env python3
"""
Min-p 结果整理脚本
将所有 minp_* 目录中的结果文件整理到指定目录中
"""

import os
import shutil
from pathlib import Path
import argparse
from datetime import datetime

def organize_minp_results(source_dir: str, target_dir: str, copy_mode: bool = False):
    """
    整理Min-p实验结果到目标目录
    
    Args:
        source_dir: 源目录路径 (包含minp_*子目录)
        target_dir: 目标目录路径
        copy_mode: 是否复制模式 (False=移动, True=复制)
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # 创建目标目录
    target_path.mkdir(parents=True, exist_ok=True)
    
    # 查找所有minp_*目录
    minp_dirs = list(source_path.glob("minp_*"))
    
    if not minp_dirs:
        print(f"在 {source_path} 中未找到任何 minp_* 目录")
        return
    
    print(f"找到 {len(minp_dirs)} 个minp目录")
    
    moved_count = 0
    failed_count = 0
    
    operation = "复制" if copy_mode else "移动"
    
    for minp_dir in minp_dirs:
        try:
            # 目标目录名（保持原名）
            target_subdir = target_path / minp_dir.name
            
            if copy_mode:
                # 复制模式
                if target_subdir.exists():
                    print(f"跳过 {minp_dir.name} (目标已存在)")
                    continue
                shutil.copytree(minp_dir, target_subdir)
                print(f"✓ 已复制: {minp_dir.name}")
            else:
                # 移动模式
                if target_subdir.exists():
                    print(f"跳过 {minp_dir.name} (目标已存在)")
                    continue
                shutil.move(str(minp_dir), str(target_subdir))
                print(f"✓ 已移动: {minp_dir.name}")
            
            moved_count += 1
            
        except Exception as e:
            print(f"✗ {operation}失败 {minp_dir.name}: {e}")
            failed_count += 1
    
    print(f"\n{operation}完成:")
    print(f"  成功: {moved_count} 个目录")
    print(f"  失败: {failed_count} 个目录")
    print(f"  目标位置: {target_path}")

def create_index_file(target_dir: str):
    """
    创建结果索引文件
    
    Args:
        target_dir: 目标目录路径
    """
    target_path = Path(target_dir)
    
    if not target_path.exists():
        print(f"目标目录不存在: {target_path}")
        return
    
    # 查找所有minp_*目录
    minp_dirs = list(target_path.glob("minp_*"))
    
    if not minp_dirs:
        print(f"在 {target_path} 中未找到任何 minp_* 目录")
        return
    
    index_data = {
        "created_at": datetime.now().isoformat(),
        "total_experiments": len(minp_dirs),
        "experiments": []
    }
    
    for minp_dir in sorted(minp_dirs):
        # 从目录名提取min-p值
        minp_str = minp_dir.name.replace("minp_", "")
        minp_values = [float(x) for x in minp_str.split("_")]
        
        # 查找结果文件
        result_files = [f for f in minp_dir.glob("*.json") 
                       if not f.name.startswith("config_")]
        
        experiment_info = {
            "directory": minp_dir.name,
            "minp_values": minp_values,
            "result_files": [f.name for f in result_files],
            "has_results": len(result_files) > 0
        }
        
        index_data["experiments"].append(experiment_info)
    
    # 保存索引文件
    index_file = target_path / "experiment_index.json"
    import json
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)
    
    print(f"📄 已创建实验索引文件: {index_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="整理Min-p实验结果")
    parser.add_argument(
        "--source",
        type=str,
        default="/cephfs/shared/sunyifan/Min-p-CoT/grid_search_results",
        help="源目录路径 (默认: grid_search_results)"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="/cephfs/shared/sunyifan/Min-p-CoT/grid_search_results/search_0.2_interval",
        help="目标目录路径 (默认: grid_search_results/search_0.2_interval)"
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="复制模式 (默认为移动模式)"
    )
    parser.add_argument(
        "--create-index",
        action="store_true",
        help="创建实验索引文件"
    )
    
    args = parser.parse_args()
    
    try:
        print("🔄 开始整理Min-p实验结果...")
        
        # 整理结果文件
        organize_minp_results(
            source_dir=args.source,
            target_dir=args.target,
            copy_mode=args.copy
        )
        
        # 创建索引文件
        if args.create_index:
            create_index_file(args.target)
        
        print("\n✅ 整理完成！")
        
    except Exception as e:
        print(f"❌ 整理失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())