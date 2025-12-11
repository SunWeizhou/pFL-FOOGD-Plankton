#!/usr/bin/env python3
"""
重命名实验图片脚本
为实验图片添加清晰的标记，便于识别
"""

import os
import shutil
from pathlib import Path

def rename_experiment_images():
    """重命名实验文件夹中的图片文件"""
    experiments_dir = Path("experiments")

    if not experiments_dir.exists():
        print("实验文件夹不存在")
        return

    # 遍历所有实验文件夹
    for exp_dir in experiments_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        exp_name = exp_dir.name  # 如: alpha0.5_with_foogd

        # 查找实验子文件夹
        for sub_dir in exp_dir.iterdir():
            if not sub_dir.is_dir():
                continue

            # 查找图片文件
            for img_file in sub_dir.glob("*.png"):
                # 获取原始文件名
                orig_name = img_file.name

                # 构建新文件名：实验名_原始文件名
                new_name = f"{exp_name}_{orig_name}"
                new_path = img_file.parent / new_name

                # 重命名文件
                print(f"重命名: {img_file} -> {new_path}")
                shutil.move(img_file, new_path)

            # 检查final_evaluation文件夹
            final_eval_dir = sub_dir / "final_evaluation"
            if final_eval_dir.exists():
                for img_file in final_eval_dir.glob("*.png"):
                    # 获取原始文件名
                    orig_name = img_file.name

                    # 构建新文件名：实验名_final_原始文件名
                    new_name = f"{exp_name}_final_{orig_name}"
                    new_path = img_file.parent / new_name

                    # 重命名文件
                    print(f"重命名: {img_file} -> {new_path}")
                    shutil.move(img_file, new_path)

def main():
    print("开始重命名实验图片...")
    rename_experiment_images()
    print("重命名完成！")

if __name__ == "__main__":
    main()