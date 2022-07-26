#!/bin/bash
# shellcheck disable=SC2034
TestDir_nlp=/home/zyj/sohu/docker/data/nlp_test
TestDir_rec=/home/zyj/sohu/docker/data/rec_test
MyImageName=yumao_10_train

cat ../data/nlp_test/train_初赛.txt ../data/nlp_test/nlp-train-dataset_复赛.txt > ../data/nlp_test/allTrain.txt

#docker run -it --gpus all -v ${TestDir_nlp}:/data/nlp ${MyImageName} python train_nlp.py --train_input /data/nlp/allTrain.txt --pretrained_path1 /data/nlp/PTM/chinese-roberta-wwm-ext/ --pretrained_path2 /data/nlp/PTM/ernie-gram-zh/ --test_input /data/nlp/nlp-test-dataset.txt --output /data/nlp/section1.txt --model_save_path /data/nlp/nlp_model/

docker run -it -v ${TestDir_rec}:/app/data/rec ${MyImageName} python train_rec.py --train_input /data/rec/rec_train.csv --test_input /data/rec/rec_test.csv --nlp_model_path /data/rec/nlp_model.pt --model_save_path /data/rec/rec_model.pt --output /data/rec/section2.txt
