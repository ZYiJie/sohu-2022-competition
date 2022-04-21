# _*_ coding:utf-8 _*_
import time
import json
import os
import argparse
import warnings

warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default="Bert", required=False)
    parser.add_argument('--train_input_path', type=str, default="data/generated_train_data.txt",
                        required=False,help="已经生成的训练集文件位置")
    parser.add_argument('--test_input_path', type=str, required=False, default="data/generated_test_data.txt",help="已经生成的测试集文件位置")
    parser.add_argument('--save_model_path', type=str, required=False, default="result/model_save",help="模型保存位置")
    parser.add_argument('--result_path', type=str, required=False, default="result/section1.txt",help="结果存放位置")
    parser.add_argument('--pretrained_path', type=str, required=False,
                        default="models/chinese_roberta_wwm_ext_pytorch/")
    parser.add_argument('--checkpoint_path', type=str, required=False, default=None,help="继续训练")
    parser.add_argument('--epochs', type=int, default=10, required=False)
    parser.add_argument('--max_sequence_input', type=int, default=256, required=False)
    parser.add_argument('--batch_size', type=int, default=64, required=False)
    parser.add_argument('--save_state', action='store_true', required=False)
    parser.add_argument('--learning_rate', type=float, default=0.0001, required=False)
    parser.add_argument('--tokenizer_path', default='models/chinese_roberta_wwm_ext_pytorch/vocab.txt', type=str,
                        required=False)
    parser.add_argument('--device', default='0', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--ratio', default=0.8, type=float, required=False, help='训练集比例')

    args = parser.parse_args()
    if args.model_type == "Bert":
        from src.modeltrain.train_bert import main
        main(args)


if __name__ == '__main__':
    main()
