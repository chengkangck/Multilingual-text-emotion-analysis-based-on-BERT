import os
import re
import logging
import argparse
import configparser
import random
from tqdm import tqdm, trange
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

config = configparser.ConfigParser()
config.read('../config.ini')

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class BertForSmooth(BertPreTrainedModel):

    def __init__(self, config, loss_weights, num_labels=72):
        super(BertForSmooth, self).__init__(config)
        self.num_labels = num_labels
        self.dropout = torch.nn.Dropout(0.2)
        self.bert = BertModel(config)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        self.loss = torch.nn.CrossEntropyLoss(torch.FloatTensor(loss_weights))
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, segment_ids, input_mask, labels=None):
        _, pooled_output = self.bert(input_ids, segment_ids, input_mask, output_all_encoded_layers=False)
        if labels is not None:
            pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            return self.loss(logits, labels)
        else:
            return logits


class InputExample(object):

    def __init__(self, sentence, label=None):
        self.sentence = sentence
        self.label = label


class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids, label=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label


class DataProcessor(object):

    @staticmethod
    def get_train_examples(data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        examples = []
        for line in open(os.path.join(data_dir, 'sentiment.train'), 'r'):
            tokens = line.strip('\n').split('\t')
            examples.append(InputExample("".join(tokens[:-1]), int(tokens[-1])))
        return examples

    @staticmethod
    def get_dev_examples(data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        examples = []
        for line in open(os.path.join(data_dir, 'sentiment.dev'), 'r'):
            tokens = line.strip('\n').split('\t')
            examples.append(InputExample("".join(tokens[:-1]), int(tokens[-1])))
        return examples

    @staticmethod
    def get_test_examples(data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        ids, examples = [], []
        for line in open(os.path.join(data_dir, 'test.data'), 'r'):
            tokens = line.strip('\n').split('\t')
            ids.append(tokens[0])
            examples.append(InputExample("".join(tokens[1:])))
        return ids, examples


def convert_examples_to_features(examples, max_seq_length, tokenizer, has_label=True):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for index, example in enumerate(examples):
        # 分类任务句首标记[CLS]
        tokens = ['[CLS]']
        if has_label:
            label = example.label
        else:
            label = None
        chars = tokenizer.tokenize(example.sentence)
        if not chars:  # 不可见字符导致返回空列表
            chars = ['[UNK]']
        tokens.extend(chars)
        if len(tokens) > max_seq_length:
            logging.debug('Example {} is too long: {}'.format(index, len(tokens)))
            tokens = tokens[0: max_seq_length]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)
        zero_padding = [0] * padding_length
        input_ids += zero_padding
        input_mask += zero_padding
        segment_ids += zero_padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label=label))
    return features


# 学习模型预热先变大然后变小
def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


def features_to_tensor(ds_features, need_labels=True):
    ds_input_ids = torch.tensor([f.input_ids for f in ds_features], dtype=torch.long)
    ds_input_mask = torch.tensor([f.input_mask for f in ds_features], dtype=torch.long)
    ds_segment_ids = torch.tensor([f.segment_ids for f in ds_features], dtype=torch.long)
    if need_labels and ds_features[0].label is not None:
        ds_labels = torch.tensor([f.label for f in ds_features], dtype=torch.long)
        ds_data = TensorDataset(ds_input_ids, ds_input_mask, ds_segment_ids, ds_labels)
    else:
        ds_data = TensorDataset(ds_input_ids, ds_input_mask, ds_segment_ids)
    return ds_data


def do_predict(dataloader, model, device):
    model.eval()
    class_probas = []
    predictions = []
    for batch in tqdm(dataloader, desc="Iteration"):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids = batch
        logits = model(input_ids, segment_ids, input_mask)
        class_proba = torch.nn.functional.softmax(logits, -1)
        class_proba = class_proba.detach().cpu().numpy()
        class_probas.extend(class_proba.tolist())
        predictions.extend(np.argmax(class_proba, -1).tolist())
    return predictions, class_probas


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model_dir", default=None, type=str, required=True,
                        help="bert pre-trained model dir")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # Optional parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_predict",
                        default=False,
                        action='store_true',
                        help="Whether to run the model in inference mode on the test set.")
    parser.add_argument("--checkpoint",
                        default=None,
                        type=str,
                        help="Specify the ckeckpoint to load.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--predict_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for predict.")
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    args = parser.parse_args()

    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
        use_gpu = True
    else:
        device = torch.device("cpu")
        use_gpu = False
    logger.info("device: {}".format(device))

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    # 统计各表情比例
    emoj = {}
    with open(os.path.join(config['path']['INPUT_PATH'], 'emoji.data'), 'r') as ef:
        for line in ef:
            emoj[line.strip().split('\t')[0]] = 0
    with open(os.path.join(config['path']['INPUT_PATH'], 'sentiment.train'), 'r') as td:
        for line in td:
            emoj[line.strip().split('\t')[1]] += 1
    esum = 0
    for key in emoj.keys():
        esum += emoj[key]
    # 根据表情比例调整损失权重
    loss_weights = []
    for key in emoj.keys():
        loss_weights.append(float(f'{emoj[key] * 1.0 / esum:.4f}'))
    loss_weights_sid = np.argsort(loss_weights)
    lwsize = len(loss_weights) - 1
    for i in range(len(loss_weights) // 2):
        loss_weights[loss_weights_sid[i]], loss_weights[loss_weights_sid[lwsize - i]]  \
            = loss_weights[loss_weights_sid[lwsize - i]], loss_weights[loss_weights_sid[i]]
    
    # 检查输出目录中是否有可加载的checkpoint并创建模型
    os.makedirs(args.output_dir, exist_ok=True)
    ckpts = [(int(filename.split('-')[1]), filename) for filename in os.listdir(args.output_dir) if re.fullmatch('checkpoint-\d+', filename)]
    ckpts = sorted(ckpts, key=lambda x: x[0])  
    if args.checkpoint or ckpts:
        if args.checkpoint:
            model_file = args.checkpoint
        else:
            # 选择global step最大的checkpoint
            model_file = os.path.join(args.output_dir, ckpts[-1][1])
        logging.info('Load %s' % model_file)
        checkpoint = torch.load(model_file, map_location='cpu')
        global_step = checkpoint['step']
        max_seq_length = checkpoint['max_seq_length']
        lower_case = checkpoint['lower_case']
        model = BertForSmooth.from_pretrained(args.bert_model_dir, state_dict=checkpoint['model_state'], loss_weights=loss_weights)
    else:
        global_step = 0
        max_seq_length = args.max_seq_length
        lower_case = args.do_lower_case
        model = BertForSmooth.from_pretrained(args.bert_model_dir, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE, loss_weights=loss_weights)

    # 分词器
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_dir, do_lower_case=lower_case)
    
    # 数据并行划分到gpu上
    if use_gpu:
        print(f'parallel data on {torch.cuda.device_count()} gpus')
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model.to(device)

    # 训练
    if args.do_train:

        if args.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps))

        args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if use_gpu:
            torch.cuda.manual_seed_all(args.seed)

        train_examples = DataProcessor.get_train_examples(args.data_dir)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

        # 优化器准备以及优化参数设置
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup_proportion,
                             t_total=num_train_steps)

        # 加载训练集数据
        train_features = convert_examples_to_features(train_examples, max_seq_length, tokenizer)
        train_data = features_to_tensor(train_features)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        # 加载验证集数据
        dev_examples = DataProcessor.get_dev_examples(args.data_dir)
        dev_features = convert_examples_to_features(dev_examples, max_seq_length, tokenizer)
        dev_data = features_to_tensor(dev_features, False)
        dev_labels = [example.label for example in dev_examples]
        dev_sampler = SequentialSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.predict_batch_size)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        sw = SummaryWriter()  # tensorboard显示数据收集
        top_ckpts = {}
        threshold = 0
        start_epoch = int(global_step / (len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps))
        residue_step = global_step % (len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.gradient_accumulation_steps
        model.train()
        for epoch in trange(start_epoch, args.num_train_epochs, desc="Epoch"):
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                if epoch == start_epoch and step <= residue_step:
                    continue 
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, labels = batch

                loss = model(input_ids, segment_ids, input_mask, labels)

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                sw.add_scalar('loss', loss.clone().cpu().data.numpy().mean(), global_step)

                # 并行划分数据到gpu的情况下, 每块gpu都会返回一个loss
                loss = loss.mean()
                loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_steps, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                # 保留验证集上f1 score最好的5个模型
                if global_step > 20000 and (step + 1) % 2000 == 0:
                    dev_predictions, _ = do_predict(dev_dataloader, model, device)
                    model.train()
                    precision, recall, f1, _ = precision_recall_fscore_support(dev_labels, dev_predictions, average='macro')
                    logger.info(f'global step: {global_step}, F1 value: {f1}')
                    sw.add_scalar('precision', precision, global_step)
                    sw.add_scalar('recall', recall, global_step)
                    sw.add_scalar('f1', f1, global_step)
                    if f1 > threshold:
                        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                        torch.save({'step': global_step, 'model_state': model_to_save.state_dict(),
                                    'max_seq_length': max_seq_length, 'lower_case': lower_case},
                                   os.path.join(args.output_dir, 'checkpoint-%d' % global_step))

                        top_ckpts[global_step] = f1
                        if len(top_ckpts) > 5:
                           sorted_ckpts = sorted(top_ckpts.items(), key=lambda x: x[1])
                           top_ckpts.pop(sorted_ckpts[0][0])
                           os.system('rm %s' % os.path.join(args.output_dir, 'checkpoint-%d' % sorted_ckpts[0][0]))
                           threshold = sorted_ckpts[1][1]

    # 预测
    if args.do_predict:
        # 加载测试集
        ids, predict_examples = DataProcessor.get_test_examples(args.data_dir)
        predict_features = convert_examples_to_features(predict_examples, max_seq_length, tokenizer, False)
        predict_data = features_to_tensor(predict_features)
        predict_sampler = SequentialSampler(predict_data)
        predict_dataloader = DataLoader(predict_data, sampler=predict_sampler, batch_size=args.predict_batch_size)

        predictions, class_probas = do_predict(predict_dataloader, model, device)

        writer = open(os.path.join(args.output_dir, "test.predict-%d" % global_step), 'w')
        writer.write('ID,Expected\n')
        for _id, label in zip(ids, predictions):
            writer.write(_id+','+str(label)+'\n')
        writer.close()


if __name__ == "__main__":
    main()
