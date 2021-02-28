import configparser
import os

config = configparser.ConfigParser()
config.read('../config.ini')

emoj = {}
with open(os.path.join(config['path']['INPUT_PATH'], 'emoji.data'), 'r') as ef:
    for line in ef:
        emoj[line.strip().split('\t')[1]] = 0

with open(os.path.join(config['path']['INPUT_PATH'], 'train.solution'), 'r') as td:
    for line in td:
        emoj[line.strip()[1:-1]] += 1

# with open(config['path']['lab'], 'a') as td:
#     td.write('\n## 表情频数统计\n')
#     for key in emoj.keys():
#         td.write(key + '\t' + str(emoj[key]) + '\n')

# with open(os.path.join(config['path']['INPUT_PATH'], ''), 'a') as td:
#     esum = 0
#     for key in emoj.keys():
#         esum += emoj[key]
    
#     for key in emoj.keys():
#         td.write(f'{emoj[key] * 1.0 / esum:.6f}')
#         td.write('\t')

esum = 0
for key in emoj.keys():
    esum += emoj[key]

for key in emoj.keys():
    print(float(f'{emoj[key] * 1.0 / esum:.4f}'))




