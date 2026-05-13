from datasets import load_dataset

# 拉取 SWE-bench Verified 测试集
print("正在下载测试集...")
dataset = load_dataset("SWE-bench/SWE-bench_Verified", split="test")

# 导出为本地 JSONL 格式，和 run.py 默认文件保持一致
output_file = "swe_bench_verified_tasks.jsonl"
dataset.to_json(output_file)
print(f"下载完成，共 {len(dataset)} 个任务，已保存为 {output_file}")
