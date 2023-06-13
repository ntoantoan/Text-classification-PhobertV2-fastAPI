import argparse


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_pretrained', default='phobertV2')
    parser.add_argument('--model_save', default='phobertV2')
    parser.add_argument('--data_path', default="datasets/dataset.csv")
    parser.add_argument("--save_path", dest="save_path", default="models")
    parser.add_argument('--max_length', default=256, type=int)
    parser.add_argument("--num_class", dest="num_class", type=int, default=3)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument("--device", dest="device", default="gpu")
    parser.add_argument("--gradient_accumulation_steps", dest="gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--boost", dest="boost", default=True, type=bool)
    parser.add_argument("--bert_finetuning", dest="bert_finetuning", default=True, type=bool)
    parser.add_argument("--dropout_p", dest="dropout_p", default=0.1, type=int)
    parser.add_argument('--batch_size', default=3, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument("--lr_main", dest="lr_main", default=1e-4, type=int)
    parser.add_argument("--early_stop", dest="early_stop", default=2, type=int)
    parser.add_argument('--epsilon', default=1e-5, type=float)
    parser.add_argument('--use_cache', action='store_true')
    parser.add_argument('--debug', action='store_true')


    args = parser.parse_args()
    return args