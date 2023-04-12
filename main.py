import argparse, os, torch
import random
import numpy as np
from utils.data import build_data
from trainer.trainer import Trainer
from models.setpred4RE import SetPred4RE


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


if __name__ == '__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    data_arg = add_argument_group('Data')

    data_arg.add_argument('--dataset_name', type=str, default="MyData")
    data_arg.add_argument('--train_file', type=str, default="./data/MyData/train.json")
    data_arg.add_argument('--valid_file', type=str, default="./data/MyData/valid.json")
    data_arg.add_argument('--test_file', type=str, default="./data/MyData/test.json")
    
    # data_arg.add_argument('--dataset_name', type=str, default="NYT-exact")
    # data_arg.add_argument('--train_file', type=str, default="./data/NYT/exact_data/train.json")
    # data_arg.add_argument('--valid_file', type=str, default="./data/NYT/exact_data/valid.json")
    # data_arg.add_argument('--test_file', type=str, default="./data/NYT/exact_data/test.json")
    
    # data_arg.add_argument('--dataset_name', type=str, default="NYT-partial")
    # data_arg.add_argument('--train_file', type=str, default="./data/NYT/casrel_data/new_train.json")
    # data_arg.add_argument('--valid_file', type=str, default="./data/NYT/casrel_data/new_valid.json")
    # data_arg.add_argument('--test_file', type=str, default="./data/NYT/casrel_data/new_test.json")
    
    # data_arg.add_argument('--dataset_name', type=str, default="WebNLG")
    # data_arg.add_argument('--train_file', type=str, default="./data/WebNLG/clean_WebNLG/new_train.json")
    # data_arg.add_argument('--valid_file', type=str, default="./data/WebNLG/clean_WebNLG/new_valid.json")
    # data_arg.add_argument('--test_file', type=str, default="./data/WebNLG/clean_WebNLG/new_test.json")

    data_arg.add_argument('--do_lower_case', type=bool, default=False)
    data_arg.add_argument('--use_bert', type=bool, default=True)
    data_arg.add_argument('--generated_data_directory', type=str, default="./data/generated_data/")
    data_arg.add_argument('--generated_param_directory', type=str, default="./data/generated_data/model_param/")
    data_arg.add_argument('--generated_result_directory', type=str, default="./data/generated_data/train_result/")
    data_arg.add_argument('--train_result_name', type=str, default="test.json")
    data_arg.add_argument('--cased_bert_directory', type=str, default="E:/PycharmProjects/bert-base-cased")
    data_arg.add_argument('--uncased_bert_directory', type=str, default="E:/PycharmProjects/bert-base-uncased")
    data_arg.add_argument("--partial", type=str2bool, default=False)
    learn_arg = add_argument_group('Learning')
    learn_arg.add_argument('--model_name', type=str, default="SPN")
    learn_arg.add_argument('--num_generated_triples', type=int, default=10) #15

    learn_arg.add_argument('--num_decoder_layers', type=int, default=3) #3

    learn_arg.add_argument('--matcher', type=str, default="avg", choices=['avg', 'min']) #avg
    learn_arg.add_argument('--na_rel_coef', type=float, default=0.25)
    learn_arg.add_argument('--rel_loss_weight', type=float, default=1.5)
    learn_arg.add_argument('--head_ent_loss_weight', type=float, default=2)
    learn_arg.add_argument('--tail_ent_loss_weight', type=float, default=2)
    learn_arg.add_argument('--fix_bert_embeddings', type=str2bool, default=True)

    learn_arg.add_argument('--batch_size', type=int, default=3)
    learn_arg.add_argument('--max_epoch', type=int, default=50)

    learn_arg.add_argument('--gradient_accumulation_steps', type=int, default=1)
    learn_arg.add_argument('--decoder_lr', type=float, default=2e-5) #2e-5
    learn_arg.add_argument('--encoder_lr', type=float, default=1e-5) #1e-5
    learn_arg.add_argument('--lr_decay', type=float, default=0.05)
    learn_arg.add_argument('--weight_decay', type=float, default=1e-5)
    learn_arg.add_argument('--max_grad_norm', type=float, default=2.5)
    learn_arg.add_argument('--optimizer', type=str, default='AdamW', choices=['Adam', 'AdamW'])
    evaluation_arg = add_argument_group('Evaluation')
    evaluation_arg.add_argument('--n_best_size', type=int, default=120)
    evaluation_arg.add_argument('--max_span_length', type=int, default=15) #NYT webNLG 10
    evaluation_arg.add_argument('--max_char_length', type=int, default=15)
    misc_arg = add_argument_group('MISC')
    misc_arg.add_argument('--refresh', type=str2bool, default=False)
    misc_arg.add_argument('--use_gpu', type=str2bool, default=True)
    misc_arg.add_argument('--visible_gpu', type=int, default=1)
    misc_arg.add_argument('--random_seed', type=int, default=1)

    args, unparsed = get_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.visible_gpu)
    for arg in vars(args):
        print(arg, ":",  getattr(args, arg))
    set_seed(args.random_seed)
    data = build_data(args)
    model = SetPred4RE(args, data.relational_alphabet.size())
    # epoch = 49
    # f1 = 0.7631
    # if args.use_gpu:
    #     model = model.cuda()
    # model.load_state_dict(torch.load(
    #     args.generated_param_directory + "%s_%s_epoch_%d_f1_%.4f.pt" % (args.model_name, args.dataset_name, epoch, f1)))
    trainer = Trainer(model, data, args)
    trainer.train_model()
