import csv
import json
import random

def convert_csv_to_json(input_file, output_file, limit=700):
    results = []
    count = 0

    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # 跳过表头

        for row in reader:
            if count >= limit:
                break

            if len(row) < 3:
                continue

            label,text= row[0], row[2]

            if label.strip().lower() in ["not_sarc", "notsarc"]:
                results.append({
                    "question": text,
                    "answer": ["no"]
                })
                count += 1
            if label.strip().lower() in ["sarc", " sarc"]:
                results.append({
                    "question": text,
                    "answer": ["yes"]
                })
                count += 1
    random.shuffle(results)

    # 随机抽取 sample_size 个
    sampled = results[:limit]

    # 保存成 JSON 文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)

    print(f"✅ Done! Saved {len(sampled)}  sarcasm samples to {output_file}")


if __name__ == "__main__":
    convert_csv_to_json("HYP-sarc-notsarc.csv", "sarc_500.json", limit=600)
