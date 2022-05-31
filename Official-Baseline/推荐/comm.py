# coding: utf-8
import os
import time
import logging 
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s" 
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT) 
logger = logging.getLogger(__file__)
import numpy as np
import pandas as pd
import random

# 存储数据的根目录
ROOT_PATH = "./data"
# 训练集
TRAIN_FILE = os.path.join(ROOT_PATH, "rec_data", "train-dataset.csv")
# 测试集
TEST_FILE = os.path.join(ROOT_PATH, "rec_data", "test-dataset.csv")

SEED = 2022

def create_dir():
    """
    创建所需要的目录
    """
    # 创建data目录
    if not os.path.exists(ROOT_PATH):
        print('Create dir: %s'%ROOT_PATH)
        os.mkdir(ROOT_PATH)
    # data目录下需要创建的子目录
    need_dirs = ["train", "evaluate", "submit", "feature", "model"]
    for need_dir in need_dirs:
        need_dir = os.path.join(ROOT_PATH, need_dir)
        if not os.path.exists(need_dir):
            print('Create dir: %s'%need_dir)
            os.mkdir(need_dir)


def check_file():
    '''
    检查数据文件是否存在
    '''
    paths = [TRAIN_FILE, TEST_FILE]
    flag = True
    not_exist_file = []
    for f in paths:
        if not os.path.exists(f):
            not_exist_file.append(f)
            flag = False
    return flag, not_exist_file


def stat_data():
    """
    统计特征最大，最小，均值
    """
    paths = [TRAIN_FILE, TEST_FILE]
    pd.set_option('display.max_columns', None)
    for path in paths:
        df = read_sample_csv(path)
        print(path + " stat: ")
        print(df.describe())
        print('Distinct count:')
        print(df.nunique())



def read_sample_csv(path):
    df = pd.read_csv(path, delimiter=",")
    df["userSeq"] = df["userSeq"].fillna("")
    df['time'] = pd.to_datetime(df['logTs'],unit='ms',origin=pd.to_datetime('1970-01-01 08:00:00'))
    df['min'] = df['time'].dt.hour*60 + df['time'].dt.minute
    df['hour'] = df['time'].dt.hour
    return df

def process_feature(df, path1, path2):
    # id_map_df = pd.read_csv("/Users/andy/Desktop/algoCompetition/dataset/hash/idMap", delimiter="\t")
    # emotion_df = pd.read_csv("/Users/andy/Desktop/algoCompetition/dataset/res_itemId.txt", delimiter="\t")
    # df_1 = pd.merge(df, id_map_df, how='left', left_on="itemId", right_on="hashMpId")
    # df_2 = pd.merge(df_1, emotion_df, how='left', on="mpId")
    # df_2["emotion"] = df_2["emotion"].fillna("-2").map(lambda x:(int)(x))
    # df_2 = df_2.drop(["hashMpId", "mpId"], axis=1)
    # return df_2
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    result=pd.concat([df,df1,df2],axis=1)
    return result


def generate_offline_sample():
    sample_path = TRAIN_FILE
    df = read_sample_csv(sample_path)

    df = process_feature(df, './data/rec_data/train-co-occurrence-feature.csv',
                                './data/rec_data/train-itemId-feature.csv')

    # valid_data_size = 200000
    # train_df = df.iloc[:-1 * valid_data_size]
    # valid_df = df.iloc[-1 * valid_data_size:]
    pvId_list = list(set(df['pvId']))
    random.shuffle(pvId_list)
    length = len(pvId_list)
    train_pvId = pvId_list[:int(length*0.95)]
    valid_pvId = pvId_list[int(length*0.95):]

    train_df = df.loc[df['pvId'].isin(train_pvId)]
    valid_df = df.loc[df['pvId'].isin(valid_pvId)]


    train_output_path = os.path.join(ROOT_PATH, "train", "sample.csv")
    valid_output_path = os.path.join(ROOT_PATH, "evaluate", "sample.csv")

    train_df.to_csv(train_output_path, index=False)
    valid_df.to_csv(valid_output_path, index=False)
    return df

def generate_submit_sample():
    input_path = TEST_FILE
    df = read_sample_csv(input_path)
    df = process_feature(df, './data/rec_data/test-co-occurrence-feature.csv',
                                './data/rec_data/test-itemId-feature.csv')
    output_path = os.path.join(ROOT_PATH, "submit", "sample.csv")
    df.to_csv(output_path, index=False)
    return df


def main():
    flag, not_exists_file = check_file()
    if not flag:
        print("请检查目录中是否存在下列文件: ", ",".join(not_exists_file))
        return
    t = time.time()
    stat_data()
    create_dir()

    logger.info("Stage: offline")
    df = generate_offline_sample()
    df.describe()

    logger.info("Stage: submit")
    df = generate_submit_sample()
    df.describe()

    print('Time cost: %.2f s'%(time.time()-t))


if __name__ == "__main__":
    main()
    