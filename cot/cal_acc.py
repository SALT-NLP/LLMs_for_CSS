import json
import numpy as np

data_dir = "experiment/multiarith_auto"

max_num = 10000000000000
total = 0
correct_list = []
with open(data_dir, "r") as rp:
    for line in rp:
        if total >= max_num: break
        line = json.loads(line)
        pred = line["pred_ans"]
        gold = line["gold_ans"]
        if line["rationale"] == "Error!":
            print("Error!")
            continue
        correct = (np.array([pred]) == np.array([gold])).sum().item()
        correct_list.append(correct)
        total += 1 #np.array([y]).size(0)
accuracy = (sum(correct_list) * 1.0 / total) * 100
print(data_dir, total, "accuracy : {}".format(accuracy))
