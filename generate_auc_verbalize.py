import json
from itertools import chain
from sklearn.metrics import roc_auc_score
import re

def normalize(text):
    return str(text).strip().lower()



# def flatten_and_normalize(answers):
#     # 如果是嵌套结构，展开并清洗
#     if any(isinstance(a, list) for a in answers):
#         answers = list(chain.from_iterable(answers))
#     return set(map(normalize, answers))

def flatten_and_normalize(answers):
    # 确保输入是 list
    if not isinstance(answers, list):
        answers = [answers]

    # 展平嵌套列表
    flat = []
    for a in answers:
        if isinstance(a, list):
            flat.extend(a)
        else:
            flat.append(a)

    # 统一转换成字符串并规范化
    return set(normalize(str(x)) for x in flat)

def answers_match(predicted, ground_truth):
    return flatten_and_normalize(predicted) == flatten_and_normalize(ground_truth)


# 路径替换成你实际的文件路径
with open("gpt35_-6 (3).json", "r") as f:
    data = json.load(f)

scores = []
labels = []
cleaned_answers = []
for item in data:
    # 取出置信分（最后的数字）
    llm_output = item.get("llm_output", "")
    try:
        score_str = llm_output.strip().split(",")[-1].strip()
        score = float(score_str) / 100.0
    except:
        score = 0.0  # fallback in case of bad format

    scores.append(score)

    #predicted = item.get("predicted_answer", [])
    predicted = item.get("llm_output", [])
    # predicted_str = predicted[3:-6]
    predicted_str = predicted[2:-6]
    predicted = predicted_str.split(", ")
    for ans in predicted:
        res = ""
        if len(ans)>0 and ans[-1]=="|":
            res = ans[:-1]
            cleaned_answers.append(res)
    
    

    ground_truth = item.get("gt_answers", [])

    label = 1 if answers_match(cleaned_answers, ground_truth) else 0
    labels.append(label)


# print("scores =", scores)
# print("labels =", labels)

auroc = roc_auc_score(labels, scores)
print(cleaned_answers)
print("AUROC:", auroc)
