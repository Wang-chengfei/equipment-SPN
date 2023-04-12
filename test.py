import argparse, os, torch
import random
import numpy as np
from utils.data import build_data
from trainer.trainer import Trainer
from models.setpred4RE import SetPred4RE
from transformers import BertTokenizer
import json


def str2bool(v):
    return v.lower() in ('true')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    return arg


def get_args():
    args, unparsed = parser.parse_known_args()
    if len(unparsed) > 1:
        print("Unparsed args: {}".format(unparsed))
    return args, unparsed


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def remove_accents(text: str) -> str:
    accents_translation_table = str.maketrans(
        "áéíóúýàèìòùỳâêîôûŷäëïöüÿñÁÉÍÓÚÝÀÈÌÒÙỲÂÊÎÔÛŶÄËÏÖÜŸ",
        "aeiouyaeiouyaeiouyaeiouynAEIOUYAEIOUYAEIOUYAEIOUY"
    )
    return text.translate(accents_translation_table)


def list_index(list1: list, list2: list) -> list:
    start = [i for i, x in enumerate(list2) if x == list1[0]]
    end = [i for i, x in enumerate(list2) if x == list1[-1]]
    if len(start) == 1 and len(end) == 1:
        return start[0], end[0]
    else:
        for i in start:
            for j in end:
                if i <= j:
                    if list2[i:j + 1] == list1:
                        index = (i, j)
                        break
        return index[0], index[1]


if __name__ == '__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    data_arg = add_argument_group('Data')

    data_arg.add_argument('--dataset_name', type=str, default="MyData")
    data_arg.add_argument('--train_file', type=str, default="./data/MyData/train.json")
    data_arg.add_argument('--valid_file', type=str, default="./data/MyData/valid.json")
    data_arg.add_argument('--test_file', type=str, default="./data/MyData/test.json")

    data_arg.add_argument('--do_lower_case', type=bool, default=False)
    data_arg.add_argument('--use_bert', type=bool, default=True)
    data_arg.add_argument('--generated_data_directory', type=str, default="./data/generated_data/")
    data_arg.add_argument('--generated_param_directory', type=str, default="./data/generated_data/model_param/")
    data_arg.add_argument('--cased_bert_directory', type=str, default="E:/PycharmProjects/bert-base-cased")
    data_arg.add_argument('--uncased_bert_directory', type=str, default="E:/PycharmProjects/bert-base-uncased")
    data_arg.add_argument("--partial", type=str2bool, default=False)
    learn_arg = add_argument_group('Learning')
    learn_arg.add_argument('--model_name', type=str, default="SPN")
    learn_arg.add_argument('--num_generated_triples', type=int, default=10)

    learn_arg.add_argument('--num_decoder_layers', type=int, default=3)

    learn_arg.add_argument('--matcher', type=str, default="avg", choices=['avg', 'min'])
    learn_arg.add_argument('--na_rel_coef', type=float, default=0.25)
    learn_arg.add_argument('--rel_loss_weight', type=float, default=1)
    learn_arg.add_argument('--head_ent_loss_weight', type=float, default=2)
    learn_arg.add_argument('--tail_ent_loss_weight', type=float, default=2)
    learn_arg.add_argument('--fix_bert_embeddings', type=str2bool, default=True)

    learn_arg.add_argument('--batch_size', type=int, default=3)
    learn_arg.add_argument('--max_epoch', type=int, default=20)

    learn_arg.add_argument('--gradient_accumulation_steps', type=int, default=1)
    learn_arg.add_argument('--decoder_lr', type=float, default=2e-5)
    learn_arg.add_argument('--encoder_lr', type=float, default=1e-5)
    learn_arg.add_argument('--lr_decay', type=float, default=0.01)
    learn_arg.add_argument('--weight_decay', type=float, default=1e-5)
    learn_arg.add_argument('--max_grad_norm', type=float, default=2.5)
    learn_arg.add_argument('--optimizer', type=str, default='AdamW', choices=['Adam', 'AdamW'])
    evaluation_arg = add_argument_group('Evaluation')
    evaluation_arg.add_argument('--n_best_size', type=int, default=100)
    evaluation_arg.add_argument('--max_span_length', type=int, default=10)  # NYT webNLG 10
    misc_arg = add_argument_group('MISC')
    misc_arg.add_argument('--refresh', type=str2bool, default=False)
    misc_arg.add_argument('--use_gpu', type=str2bool, default=True)
    misc_arg.add_argument('--visible_gpu', type=int, default=1)
    misc_arg.add_argument('--random_seed', type=int, default=1)

    args, unparsed = get_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.visible_gpu)
    data = build_data(args)
    relational_alphabet = data.relational_alphabet
    # 加载模型
    epoch = 2
    f1 = 0.7479
    model = SetPred4RE(args, relational_alphabet.size())
    if args.use_gpu:
        model = model.cuda()
    model.load_state_dict(torch.load(
        args.generated_param_directory + "%s_%s_epoch_%d_f1_%.4f.pt" % (args.model_name, args.dataset_name, epoch, f1)))

    # 加载tokenizer
    if args.do_lower_case:
        tokenizer = BertTokenizer.from_pretrained(args.uncased_bert_directory, do_lower_case=True)
    else:
        tokenizer = BertTokenizer.from_pretrained(args.cased_bert_directory, do_lower_case=False)

    # 预处理数据
    input_doc = "./data/MyData/valid.json"
    output_doc = "./data/MyData/valid_res.json"
    with open(input_doc, "r", encoding="utf8") as fp:
        datas = json.load(fp)
    lines = []
    for data in datas:
        lines.append(data["sentText"])
    # lines = ["The 10TP was a Polish light cruiser tank that never left the prototype status .",
    #          "Didgori-3 , like two previous Didgori Armoured Personnel Carrier and Didgori-2 is equipped with night/thermal imaging cameras and GPS navigation system .",
    #          "The adoption of PLZ-05 signified China 's paradigm shift in artillery doctrines , moving from the Soviet model to Western model .",
    #          "Discoveries made during testing led to the design phase of the newer 14TP model , which was never completed due to the onset of World War II .",
    #          "The 10TP prototype itself was of an original design implementing some general ideas suggested by John Walter Christie but also many new technical solutions .",
    #          "At the end of the 1920s , the Polish Armed Forces felt that they needed a new tank model ."]
    samples = []
    for i in range(len(lines)):
        token_sent = [tokenizer.cls_token] + tokenizer.tokenize(remove_accents(lines[i])) + [tokenizer.sep_token]
        target = {"relation": [], "head_start_index": [], "head_end_index": [], "tail_start_index": [],
                  "tail_end_index": []}
        sent_id = tokenizer.convert_tokens_to_ids(token_sent)
        samples.append([i, sent_id, target])

    # 批处理
    batch_size = args.batch_size
    eval_num = len(samples)
    total_batch = eval_num // batch_size + 1
    prediction = dict()

    for batch_id in range(total_batch):
        start = batch_id * batch_size
        end = (batch_id + 1) * batch_size
        if end > eval_num:
            end = eval_num
        eval_instance = samples[start:end]
        if not eval_instance:
            continue

        # 模型预测
        input_ids, attention_mask, target, info = model.batchify(eval_instance)
        gen_triples = model.gen_triples(input_ids, attention_mask, info)
        prediction.update(gen_triples)

    # 还原预测结果
    # threshold = -0.1
    result_list = []
    for i in range(len(prediction)):
        result = dict()
        sent_ids = samples[i][1]
        sent = lines[i]
        result["sentText"] = sent
        result["relationMentions"] = []
        triples = prediction[i]
        for triple in triples:
            label = relational_alphabet.instances[triple.pred_rel]
            em1Text = sent_ids[triple.head_start_index:triple.head_end_index + 1]
            em1Text = tokenizer.convert_ids_to_tokens(em1Text)
            em1Text = tokenizer.convert_tokens_to_string(em1Text)
            em2Text = sent_ids[triple.tail_start_index:triple.tail_end_index + 1]
            em2Text = tokenizer.convert_ids_to_tokens(em2Text)
            em2Text = tokenizer.convert_tokens_to_string(em2Text)
            relationMention = dict()
            relationMention["em1Text"] = em1Text
            relationMention["em2Text"] = em2Text
            relationMention["label"] = label
            result["relationMentions"].append(relationMention)
        result_list.append(result)
        print(result["sentText"])
        print(result["relationMentions"])
        print()

    # 写入json文件
    with open(output_doc, 'w') as f:
        json.dump(result_list, f)
