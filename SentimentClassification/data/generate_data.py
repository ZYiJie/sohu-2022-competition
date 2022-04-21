# _*_ coding:utf-8 _*_
import json
import os.path
import pandas as pd


def new_data_piece(id: int, content: str, entity: dict,
                   label: int,is_train:bool) -> dict:
    if is_train:
        return {
            "id": id,
            "content": content,
            "entity": entity,
            "label": label,
        }
    else:
        return {
            "id": id,
            "content": content,
            "entity": entity
        }

if __name__ == '__main__':
    #
    input_file = 'raw_data/'
    output_file = './'

    #train_generate_data
    with open(os.path.join(input_file,"train.txt"), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        with open(os.path.join(output_file,"generated_train_data.txt"), 'w',encoding='utf-8') as fw:
            for line in lines:
                taskData = json.loads(line.strip())
                contents = taskData['content']
                entitys = taskData['entity']
                id = taskData["id"]
                for entity, label in entitys.items():
                    data = new_data_piece(id, content=contents,
                                          entity=entity, label=int(label),is_train=True)
                    fw.write(json.dumps(data, ensure_ascii=False) + "\n")

    #test_generate_data
    with open(os.path.join(input_file,"test.txt"), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        with open(os.path.join(output_file,"generated_test_data.txt"), 'w',encoding='utf-8') as fw:
            for line in lines:
                taskData = json.loads(line.strip())
                contents = taskData['content']
                entitys = taskData['entity']
                id = taskData["id"]
                for entity in entitys:
                    data = new_data_piece(id, content=contents,
                                          entity=entity, label=0,is_train=False)
                    fw.write(json.dumps(data, ensure_ascii=False) + "\n")