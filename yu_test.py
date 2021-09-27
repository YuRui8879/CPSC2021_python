import os
from entry_2021 import challenge_entry
from utils import save_dict
from score_2021 import score
import numpy as np
from distutils import dir_util

pic_path = r'.\pic'
if os.path.exists(pic_path):
    dir_util.remove_tree(pic_path)

DATA_PATH = r'C:\Users\yurui\Desktop\item\cpsc\data\all_data'
RESULT_PATH = r'C:\Users\yurui\Desktop\item\cpsc\code\pretrain\out'
RECORDS_PATH = r'C:\Users\yurui\Desktop\item\cpsc\code\pretrain\test_record'
if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH)

# test_set = open(os.path.join(DATA_PATH, 'RECORDS'), 'r').read().splitlines()
test_set = open(os.path.join(RECORDS_PATH, 'RECORDS'), 'r').read().splitlines()
for i, sample in enumerate(test_set):
    print(sample)
    sample_path = os.path.join(DATA_PATH, sample)
    pred_dict = challenge_entry(sample_path)

    save_dict(os.path.join(RESULT_PATH, sample+'.json'), pred_dict)

score_avg = score(DATA_PATH, RESULT_PATH)
print('AF Endpoints Detection Performance: %0.4f' %score_avg)

with open(os.path.join(RESULT_PATH, 'score.txt'), 'w') as score_file:
    print('AF Endpoints Detection Performance: %0.4f' %score_avg, file=score_file)

    score_file.close()

try:
    file_path = r'.\pic\confusion_matrix.npy'
    confusion_matrix = np.load(file_path)
    print(confusion_matrix)
except:
    print('no_confusion_matrix.npy')