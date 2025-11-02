import csv
import json

# Step 1: 加载 emotion.txt，构建索引 → 情绪标签映射
emotion_file = "emotions.txt"
emotion_labels = []
with open(emotion_file, 'r', encoding='utf-8') as f:
    emotion_labels = [line.strip() for line in f if line.strip()]

# Step 2: 读取 TSV 文件
input_tsv = "dev.tsv"  # 替换为你的真实文件名
output_json = "converted_output_with_labels.json"

data = []

with open(input_tsv, newline='', encoding='utf-8') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    for row in reader:
        if len(row) < 2:
            continue

        question = row[0].strip()
        index_strs = row[1].strip().split(',')

        # 将 index 转为情绪标签
        labels = []
        for idx in index_strs:
            idx = idx.strip()
            if idx.isdigit():
                i = int(idx) - 1  # index 从 1 开始
                if 0 <= i < len(emotion_labels):
                    labels.append(emotion_labels[i])

        data.append({
            "question": question,
            "answer": labels
        })

# Step 3: 写入 JSON 文件
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"成功写入 JSON 文件：{output_json}")
