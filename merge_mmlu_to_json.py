import csv
import json
from collections import defaultdict, OrderedDict

input_file = "raw_datasets/train.csv"        # 输入文件名
output_file = "datasets/mmlu_merged.json"  # 输出文件名

data_dict = defaultdict(lambda: {
    "question": "",
    "options": {},   # 存 A,B,C,D->内容
    "answers": set()
})

# 读取 CSV
cnt = 0
with open(input_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        cnt+=1
        if cnt>800:
            continue
        q = row["prompt"].strip()
        data_dict[q]["question"] = q
        # 保存选项文本
        for opt in ["A", "B", "C", "D"]:
            if opt in row and row[opt].strip():
                data_dict[q]["options"][opt] = row[opt].strip()
        # 收集正确答案字母
        ans = row["answer"].strip().upper()
        if ans in data_dict[q]["options"]:
            data_dict[q]["answers"].add(ans)

# 生成输出列表
output = []
for q, item in data_dict.items():
    opts = item["options"]
    ans_letters = sorted(list(item["answers"]))
    ans_texts = [opts[a] for a in ans_letters if a in opts]

    # 单选题 -> 字符串，多选题 -> 列表
    # if len(ans_texts) == 1:
    #     answer_out = ans_texts[0]
    # else:
    if len(ans_letters) == 1:
        answer_out = ans_letters[0]
    else:
        answer_out = ans_letters
    

    output.append(OrderedDict([
        ("question", item["question"]),
        ("answer", answer_out)
    ]))

# 写出 JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=4)

print(f"✅ 转换完成，共 {len(output)} 道题，输出文件: {output_file}")
