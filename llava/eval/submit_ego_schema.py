import json
import pandas as pd
import glob

def fuzzy_matching(pred):
    return pred.split(' ')[0].rstrip('.').strip()

data_root = 'results/BIMBA-LLaVA-Qwen2-7B/EgoSchema'

submission = {"q_uid": [], "answer": []}
for file in glob.glob(f"{data_root}/outputs/*.json"):
    with open(file, "r") as f:
        sample = json.load(f)
    submission["q_uid"].append(sample["id"])
    pred = fuzzy_matching(sample["prediction"])
    pred = ord(pred) - ord('A')
    submission["answer"].append(pred)

print(len(submission["q_uid"]), len(submission["answer"]))

df = pd.DataFrame(submission)
df.to_csv(f"{data_root}/es_submission.csv", index=False)

# kaggle competitions submit -c egoschema-public -f results/BIMBA-LLaVA-Qwen2-7B/EgoSchema/es_submission.csv -m "BIMBA-LLaVA-Qwen2-7B"
