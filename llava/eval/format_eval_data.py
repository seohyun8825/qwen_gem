import json
import os
import glob
from decord import VideoReader, cpu
import numpy as np
import pandas as pd
import random

#Format dataset in a json file so that each sample has the following:
# (id, video_id, video, question_type, question, candidates, answer)

#VideoMME
df = pd.read_parquet("DATAS/eval/VideoMME/test-00000-of-00001.parquet")
#Columns: [video_id, duration, domain, sub_category, url, videoID, question_id, task_type, question, options, answer]

dataset = []
for index, row in df.iterrows():
    sample = {}
    sample['id'] = index 
    sample['video_id'] = row['video_id']
    sample['video'] = row['videoID'] + '.mp4'
    sample['question_type'] = row['duration']
    sample['question'] = row['question']
    sample['candidates'] = row['options'].tolist()
    sample['answer'] = row['answer']
    dataset.append(sample)

with open("DATAS/eval/VideoMME/formatted_dataset.json", "w") as f:
    json.dump(dataset, f, indent=4)


