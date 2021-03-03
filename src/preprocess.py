import random
import configparser
import os

config = configparser.ConfigParser()
config.read('../config.ini')

emoj = {}
with open(os.path.join(config['path']['INPUT_PATH'], 'emoji.data'), 'r') as ef:
    for line in ef:
        items = line.strip().split('\t')
        emoj[items[1]] = items[0]

dataset = [[] for _ in emoj]
labels = []
with open(os.path.join(config['path']['INPUT_PATH'], 'train.solution'), 'r') as ts:
    for line in ts:
        labels.append(line.strip()[1:-1])
with open(os.path.join(config['path']['INPUT_PATH'], 'train.data'), 'r') as td:
    for cnt, line in enumerate(td):
        dataset[int(emoj[labels[cnt]])].append([line.strip(), emoj[labels[cnt]]])

dev_content = []
for item_ in dataset:
    random.shuffle(item_)
    dev_content += item_[:5]
random.shuffle(dev_content)

senti_train = []
for item__ in dataset:
    senti_train += item__[5:]
random.shuffle(senti_train)
with open(os.path.join(config['path']['INPUT_PATH'], 'sentiment.train'), 'w') as st:
    for item in senti_train:
        st.write(item[0] + '\t' + item[1] + '\n')

with open(os.path.join(config['path']['INPUT_PATH'], 'sentiment.dev'), 'w') as sd:
    for item in dev_content:
        sd.write(item[0] + '\t' + item[1] + '\n')
