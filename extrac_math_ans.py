import json
import re

input_file = "raw_datasets/train.jsonl"    # 输入文件（每行一个JSON对象）
output_file = "datasets/math_cleaned.json"    # 输出文件
max_count = 500                      # 只取前500条

data = []
with open(input_file, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= max_count:
            break
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        
        question = item.get("question", "").strip()
        answer_raw = item.get("answer", "").strip()
        
        # 提取 "####" 后的数字或小数
        match = re.search(r"####\s*([\-]?\d+(\.\d+)?)", answer_raw)
        if match:
            final_answer = match.group(1)
        else:
            # 没有 #### 时，尝试匹配最后一个数字
            nums = re.findall(r"[\-]?\d+(?:\.\d+)?", answer_raw)
            final_answer = nums[-1] if nums else ""

        data.append({
            "question": question,
            "answer": final_answer
        })

# 写入 JSON 数组文件
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"✅ 已提取 {len(data)} 条记录，输出文件: {output_file}")
