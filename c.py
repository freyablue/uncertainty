import json
from collections import Counter
from pathlib import Path
def measure_consistency_from_files(json_paths):
    answer_tuples = []
    
    for path in json_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            pre_lst = []
            for i in range(len(data)):
                pre_lst.append(data[i]["predicted_answer"])
        # 提取 predicted_answer
            # 转成 tuple 便于哈希
            answer_tuples.append(pre_lst)
    sim_lst = []
    ans_num, sim_count = len(answer_tuples[0]),len(answer_tuples)
    print(ans_num,sim_count)
    curr_score =[]
    for i in range(ans_num):
        now_score = 0
        for m in range(sim_count-1):
            for n in range(m+1,sim_count):
                print(answer_tuples[m][i],answer_tuples[n][i])
                now_score+=(answer_tuples[m][i]==answer_tuples[n][i])
        now_score *= 2/((sim_count)*(sim_count-1))
        curr_score.append(now_score)
    print(curr_score)
                
        

    return sum(curr_score)/len(curr_score)

    # 假设你的 JSON 文件都在某个目录里，比如 ./results
dir_path = Path("./results")

json_files = []
for file_path in dir_path.iterdir():
    if file_path.is_file():
        json_files.append(str(file_path.resolve()))
print(json_files)

print(measure_consistency_from_files(json_files))
