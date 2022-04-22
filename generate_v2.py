import transformers
import torch
import os
import json
import random
import numpy as np
import argparse

from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel

from transformers import GPT2LMHeadModel
from transformers import BertTokenizer
from os.path import join, exists
from itertools import zip_longest, chain
# from chatbot.model import DialogueGPT2Model
from rouge import Rouge
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F

PAD = '[PAD]'
pad_id = 0


def set_interact_args():
    """
    Sets up the training arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1', type=str, required=False, help='生成设备')
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成的temperature')
    parser.add_argument('--topk', default=8, type=int, required=False, help='最高k选1')
    parser.add_argument('--topp', default=0.7, type=float, required=False, help='最高积累概率')
    parser.add_argument('--model_config', default='summary_model/config.json', type=str, required=False,
                        help='模型参数')
    parser.add_argument('--log_path', default='data/interacting.log', type=str, required=False, help='interact日志存放位置')
    parser.add_argument('--voca_path', default='vocabulary/vocab_small.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--dialogue_model_path', default='model/checkpoint-8', type=str, required=False, help='对话模型路径')
    parser.add_argument('--save_samples_path', default="sample/", type=str, required=False, help="保存聊天记录的文件路径")
    parser.add_argument('--repetition_penalty', default=1.1, type=float, required=False,
                        help="重复惩罚参数，若生成的对话重复性较高，可适当提高该参数")
    parser.add_argument('--seed', type=int, default=66, help='设置种子用于生成随机数，以使得训练的结果是确定的')
    parser.add_argument('--max_len', type=int, default=400, help='每个utterance的最大长度,超过指定长度则进行截断')
    parser.add_argument('--max_history_len', type=int, default=1, help="dialogue history的最大长度")
    parser.add_argument('--no_cuda', default=False, help='不使用GPU进行预测')
    return parser.parse_args()


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        # ...表示其他维度由计算机自行推断
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # 对logits进行递减排序
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def main():
    args = set_interact_args()

     # 固定隨機種子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # 当用户使用GPU,并且GPU可用时
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    # args.cuda = False
    device = 'cuda'
    #logger.info('using device:{}'.format(device))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = GPT2LMHeadModel.from_pretrained(args.dialogue_model_path)
    model.to(device)
    model.eval()
    rouge = Rouge()
    print('***********************Summary model start************************')

    while True:
        try:
            print("請輸入文章: ")
            text = input()
            article = text
            max_score = -1
            for i in tqdm(range(15)):
                if len(article) : text = article[:1000]
                input_ids = [tokenizer.cls_token_id]  # 每个input以[CLS]为开头                
                input_ids.extend(tokenizer.encode(text, max_length = 1024, truncation = True))
                input_ids.append(tokenizer.sep_token_id)
                curr_input_tensor = torch.tensor(input_ids).long().to(device)
                #print(curr_input_tensor)
                generated = []
                # 最多生成max_len个token
                for _ in range(args.max_len):
                    outputs = model(input_ids=curr_input_tensor)
                    next_token_logits = outputs[0][-1, :]
                    # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
                    for id in set(generated):
                        next_token_logits[id] /= args.repetition_penalty
                    next_token_logits = next_token_logits / args.temperature
                    # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能>是[UNK]这个token
                    next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
                    filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=args.topk, top_p=args.topp)
                    # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
                    next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                    if next_token == tokenizer.sep_token_id:  # 遇到[SEP]则表>明response生成结束
                        break
                    generated.append(next_token.item())
                    curr_input_tensor = torch.cat((curr_input_tensor, next_token), dim=0)

                text = tokenizer.convert_ids_to_tokens(generated)
                summary ="".join(text)
                #print(summary)
                score = rouge.get_scores(' '.join(list(summary)), ' '.join(list(article)))
                if score[0]["rouge-l"]["f"] > 0.2 or score[0]["rouge-l"]["p"] > 0.3:
                    score_all = score[0]["rouge-l"]["f"]*0.1 + score[0]["rouge-l"]["p"]*0.4 + score[0]["rouge-2"]["p"]*0.4

                    if score_all > max_score:
                        #print(f"!!!!!!!!score: {score_all}")
                        max_score = score_all
                        best_summary = summary
                        best_score = max_score
                    #print(f"summary:  {summary}")
                    #print("="*20)
                    #print(score)
                    #print("="*20)
            print(f"Best: {best_summary}")
            #print(best_score)
            #df_test.loc[num_arti, "auto_summary"] = best_summary
        except KeyboardInterrupt:
            break


if __name__ == '__main__':
    main()

