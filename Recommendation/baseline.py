# coding: utf-8

import os
import shutil
import sys
import time
import pandas as pd
import tensorflow as tf
if tf.__version__ >= '2.0':
  tf = tf.compat.v1
from tensorflow import feature_column as fc

import comm
import evaluation

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 1024, 'batch_size')
flags.DEFINE_integer('embed_dim', 32, 'embed_dim')
flags.DEFINE_integer('num_epoch', 1, 'num_epoch')
flags.DEFINE_float('learning_rate', 0.001, 'learning_rate')
flags.DEFINE_float('embed_l2', None, 'embedding l2 reg')

SEED = 2022

# 存储数据的根目录
ROOT_PATH = "./data"
# 训练集
TRAIN_FILE = os.path.join(ROOT_PATH, "train/sample.csv")
# 测试集
TEST_FILE = os.path.join(ROOT_PATH, "evaluate/sample.csv")
MODEL_FILE = os.path.join(ROOT_PATH, "model")


class DeepModel(object):

    def __init__(self, dnn_feature_columns):
        super(DeepModel, self).__init__()
        self.estimator = None
        self.dnn_feature_columns = dnn_feature_columns
        tf.logging.set_verbosity(tf.logging.INFO)

    def build_estimator(self):
        model_dir = MODEL_FILE
        if not os.path.exists(model_dir):
            # 如果模型目录不存在，则创建该目录
            os.makedirs(model_dir)
        # 训练时如果模型目录已存在，则清空目录
        shutil.rmtree(model_dir)

        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta1=0.9, beta2=0.999,
                                           epsilon=1)
        config = tf.estimator.RunConfig(model_dir=model_dir, tf_random_seed=SEED)
        self.estimator = tf.estimator.DNNClassifier(
            model_dir=model_dir,
            feature_columns=self.dnn_feature_columns,
            hidden_units=[32, 8],
            optimizer=optimizer,
            config=config)


    def file_to_dataset(self, file_path, isTest, shuffle, batch_size, num_epochs):
        if (isTest):
            label_name = None
        else:
            label_name = "label"


        ds = tf.data.experimental.make_csv_dataset(
            file_path,
            batch_size=batch_size,  # 为了示例更容易展示，手动设置较小的值
            label_name=label_name,
            shuffle = shuffle,
            shuffle_seed = SEED,
            num_epochs = num_epochs,
            prefetch_buffer_size = tf.data.experimental.AUTOTUNE,

        )
        return ds

    def input_fn_train(self, path):
        return self.file_to_dataset(path, isTest=False, shuffle=True, batch_size=FLAGS.batch_size,
                                  num_epochs=FLAGS.num_epoch)

    def input_fn_evaluate(self, path):
        return self.file_to_dataset(path, isTest=False, shuffle=False, batch_size=FLAGS.batch_size * 10, num_epochs=1)

    def input_fn_predict(self, path):
        return self.file_to_dataset(path, isTest=True, shuffle=False, batch_size=FLAGS.batch_size * 10, num_epochs=1)

    def train(self):
        self.estimator.train(
            input_fn=lambda: self.input_fn_train(TRAIN_FILE)
        )

    def evaluate(self):
        input_path = TEST_FILE
        df = pd.read_csv(input_path, delimiter=",", usecols=["pvId", "label"])
        pvid_list = df['pvId'].astype(str).tolist()
        predicts = self.estimator.predict(
            input_fn=lambda: self.input_fn_evaluate(input_path)
        )
        predicts_df = pd.DataFrame.from_dict(predicts)
        logits = predicts_df["logistic"].map(lambda x: x[0])
        labels = df["label"].values
        gauc = evaluation.gAUC(labels, logits, pvid_list)
        return gauc

    
    def predict(self):
        submit_dir = os.path.join(ROOT_PATH, "submit")
        input_path = os.path.join(submit_dir, "sample.csv")
        id_df = pd.read_csv(input_path, delimiter=",", usecols=["testSampleId"])
        t = time.time()
        predicts = self.estimator.predict(
            input_fn=lambda: self.input_fn_predict(input_path)
        )
        predicts_df = pd.DataFrame.from_dict(predicts)
        logits = predicts_df["logistic"].map(lambda x: x[0])
        # 计算2000条样本平均预测耗时（毫秒）
        ts = (time.time()-t)*1000.0/len(id_df)*2000.0
        output_df = pd.concat([id_df, logits], axis=1)
        output_path = os.path.join(submit_dir, "section2.txt")
        output_df.to_csv(output_path, index=False,sep="\t", header=["Id", "result"])
        print("predict耗时(毫秒)=%s" %ts)
        return output_df, ts


def get_feature_columns():
    feature_columns = list()
    user_cate = fc.categorical_column_with_hash_bucket("suv", 40000)
    feed_cate = fc.categorical_column_with_hash_bucket("itemId", 4000, tf.int64)
    city_cate = fc.categorical_column_with_hash_bucket("city", 200, tf.int64)
    user_embedding = fc.embedding_column(user_cate, FLAGS.embed_dim, max_norm=FLAGS.embed_l2)
    feed_embedding = fc.embedding_column(feed_cate, FLAGS.embed_dim, max_norm=FLAGS.embed_l2)
    city_cate_embedding = fc.embedding_column(city_cate, FLAGS.embed_dim, max_norm=FLAGS.embed_l2)
    feature_columns.append(user_embedding)
    feature_columns.append(feed_embedding)
    feature_columns.append(city_cate_embedding)

    # emotion_cate = fc.categorical_column_with_hash_bucket("emotion", 100, tf.int64)
    # emotion_embedding = fc.embedding_column(emotion_cate, FLAGS.embed_dim, max_norm=FLAGS.embed_l2)
    # feature_columns.append(emotion_embedding)
    return feature_columns


def main(argv):
    # 检查gpu状态
    print("是否有可用GPU:", tf.test.is_gpu_available())

    t = time.time() 
    feature_columns = get_feature_columns()

    model = DeepModel(feature_columns)
    model.build_estimator()

    # 训练
    model.train()
    print("finish train!")
    # 计算验证集AUC
    evalGAUC = model.evaluate()
    print("evaluation GAUC = %s" % evalGAUC)
    # 生成submit文件
    res = model.predict()

    print('Time cost: %.2f s'%(time.time()-t))

    # evaluation.cal_score()
    sys.exit(0)


if __name__ == "__main__":
    tf.app.run(main)
    