# -*- coding:utf-8 -*-
"""
Seamese architecture+abcnn
"""
from __future__ import division
import random
import os
import time
import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
from keras.utils import to_categorical
import tensorflow as tf
FLAGS = tf.flags.FLAGS
from tensorflow.contrib import learn
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib import rnn
from nltk.stem import SnowballStemmer
import re
import jieba
from string import punctuation
random.seed(2018)
np.random.seed(2018)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Data loading path
# tf.flags.DEFINE_string("train_data_file", "H:/tb/project0/quora/quora_duplicate_questions.tsv", "train data path.")
# tf.flags.DEFINE_string("model_data_path", "H:/tb/project0/quora/model/", "model path for storing.")
# tf.flags.DEFINE_string("train_data_file", "E:/data/quora-duplicate/train.tsv", "train data path.")
tf.flags.DEFINE_string("train_data_file", "D:/DF/sentence_theme_based_sentiment/data/train.csv", "train data path.")
tf.flags.DEFINE_string("dictionary", "./utils/dictionary.txt", "dictionary path.")
tf.flags.DEFINE_string("stoplist", "./utils/stoplist.txt", "stoplist path.")
tf.flags.DEFINE_string("model_data_path", "D:/DF/sentence_theme_based_sentiment/model/", "model path for storing.")

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("subject_class", 10, "number of classes (default: 2)")
tf.flags.DEFINE_integer("sentiment_class", 3, "number of classes (default: 2)")
tf.flags.DEFINE_float("lr", 0.002, "learning rate (default: 0.002)")
tf.flags.DEFINE_integer("embedding_dim", 150, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("sentence_len", 30, "Maximum length for sentence pair (default: 50)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# LSTM Hyperparameters
tf.flags.DEFINE_integer("hidden_dim", 128, "Number of filters per filter size (default: 128)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 300, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 300, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 300, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("last_layer", 'FC', "Use FC or GAP as the last layer")


class Utils:
    @staticmethod
    def evaluation(y_true, y_predict):
        accuracy = accuracy_score(y_true, y_predict)
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_predict)
        print('accuracy:' + str(accuracy))
        print('precision:' + str(precision))
        print('recall:' + str(recall))
        print('f1:' + str(f1))

    def show_model_effect(self, history, model_path):
        """将训练过程中的评估指标变化可视化"""

        # summarize history for accuracy
        plt.plot(history.history["acc"])
        plt.plot(history.history["val_acc"])
        plt.title("Model accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
        plt.savefig(model_path+"/Performance_accuracy.jpg")

        # summarize history for loss
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("Model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
        plt.savefig(model_path+"/Performance_loss.jpg")


class DataHelpers:
    def flatten(self, l):
        return [item for sublist in l for item in sublist]

    def data_cleaning(self, text, remove_stop_words=False):
        # Clean the text, with the option to remove stop_words and to stem words.
        stop_words = [' ', '我', '你', '还', '会', '因为', '所以', '这', '是', '和',
                      '了', '的', '也', '哦']

        # Clean the text
        text = re.sub(r"[0-9]", " ", text)

        # Remove punctuation from text
        # text = ''.join([c for c in text if c not in punctuation])

        # Optionally, remove stop words
        if remove_stop_words:
            text = text.split()
            text = [w for w in text if not w in stop_words]
            text = " ".join(text)

        # Return a list of words
        return text

    def process_questions(self, question_list, df):
        '''transform questions and display progress'''
        for question in df['sentence_seq']:
            question_list.append(self.text_to_wordlist(question, remove_stop_words=False))
            if len(question_list) % 1000 == 0:
                progress = len(question_list) / len(df) * 100
                print("{} is {}% complete.".format('sentence sequence ', round(progress, 1)))
        return question_list

    def sentence_cut(self, data, dict=True):
        sentence_seq = []

        if dict:
            jieba.load_userdict(FLAGS.dictionary)

        for sentence in data['content']:
            seg_list = jieba.cut(sentence, cut_all=False)
            # print("Default Mode: " + "/ ".join(seg_list))  # 精确模式
            sentence_seg = ' '.join(seg_list)
            sentence_clean = self.data_cleaning(sentence_seg, remove_stop_words=True)
            # print(sentence_clean)
            sentence_seq.append(sentence_clean)
            if len(sentence_seq) % 1000 == 0:
                progress = len(sentence_seq) / len(data) * 100
                print("{} is {}% complete.".format('sentence sequence ', round(progress, 1)))

        data['sentence_seq'] = sentence_seq
        # print(data['sentence_seq'])

        return data

    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]


class ABCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(self, sequence_length_left, sequence_length_right, num_classes, vocab_size, embedding_size, l2_reg_lambda=0.0):
        self.sequence_length_left = sequence_length_left
        self.sequence_length_right = sequence_length_right
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.l2_reg_lambda = l2_reg_lambda

        self.set_placeholder()
        l2_loss = tf.constant(0.0)

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0), name="W_emb")
            self.embedded_chars_left = tf.nn.embedding_lookup(self.W, self.input_left)
            self.embedded_chars_expanded_left = tf.expand_dims(self.embedded_chars_left, -1)  # [batch, s, d, 1]

            self.embedded_chars_right = tf.nn.embedding_lookup(self.W, self.input_right)
            self.embedded_chars_expanded_right = tf.expand_dims(self.embedded_chars_right, -1)
        print(self.embedded_chars_expanded_right)

        channel_list = [1, 64, 128]
        filters_list = [64, 128, 64]
        abcnn1 = [True, False, False]
        abcnn2 = [False, False, False]
        branch_am_cnn_left, branch_am_cnn_right = \
            self.branch_am_cnn(self.embedded_chars_expanded_left, self.embedded_chars_expanded_right,
                               channel=(2 if abcnn1[0] else 1),
                               width=self.embedding_size, filter_size=3, num_filters=filters_list[0], conv_pad='VALID',
                               pool_pad='VALID', name='conv_1', left_len=self.sequence_length_left,
                               right_len=self.sequence_length_right, abcnn1=abcnn1[0], abcnn2=abcnn2[0])
        branch_am_cnn_left, branch_am_cnn_right = \
            self.branch_am_cnn(branch_am_cnn_left, branch_am_cnn_right,
                               channel=(2 if abcnn1[1] else 1),
                               width=channel_list[1], filter_size=3, num_filters=filters_list[1], conv_pad='VALID',
                               pool_pad='VALID', name='conv_2', left_len=self.sequence_length_left,
                               right_len=self.sequence_length_right,
                               abcnn1=abcnn1[1], abcnn2=abcnn2[1])
        branch_am_cnn_left, branch_am_cnn_right = \
            self.branch_am_cnn(branch_am_cnn_left, branch_am_cnn_right,
                               channel=(2 if abcnn1[2] else 1),
                               width=channel_list[2], filter_size=3, num_filters=filters_list[2], conv_pad='VALID',
                               pool_pad='VALID', name='conv_3', left_len=self.sequence_length_left,
                               right_len=self.sequence_length_right,
                               abcnn1=abcnn1[2], abcnn2=abcnn2[2])

        with tf.name_scope("output"):
            gap_pool_left = tf.nn.avg_pool(branch_am_cnn_left, ksize=[1, FLAGS.sentence_len, 1, 1], strides=[1, FLAGS.sentence_len, 1, 1], padding='SAME')
            gap_pool_right = tf.nn.avg_pool(branch_am_cnn_right, ksize=[1, FLAGS.sentence_len, 1, 1], strides=[1, FLAGS.sentence_len, 1, 1], padding='SAME')
            print(gap_pool_right)
            pool_output = tf.concat([gap_pool_left, gap_pool_right], 2)
            print(pool_output)
            pool_output = tf.reduce_mean(pool_output, axis=[1, 3])
            print(pool_output)

            W_o = tf.get_variable("W_o", shape=[filters_list[2]*2, FLAGS.num_class], initializer=tf.contrib.layers.xavier_initializer())
            b_o = tf.Variable(tf.constant(0.1, shape=[2]), name="b_o")
            l2_loss += tf.nn.l2_loss(W_o)
            l2_loss += tf.nn.l2_loss(b_o)
            self.scores_o = tf.nn.xw_plus_b(pool_output, W_o, b_o, name="scores_o")
            self.scores_o = tf.nn.sigmoid(self.scores_o)
            self.predictions = tf.argmax(self.scores_o, 1, name="predictions")
            print(self.scores_o)

        # if FLAGS.last_layer == 'GAP':
        #     # use GAP for softmax
        #     with tf.name_scope("GAP1"):
        #         filter_shape = [1, filters_list[2], 2, FLAGS.num_class]
        #         W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W_GAP')
        #         # b = tf.Variable(tf.constant(0.1, shape=[2]), name='b_GAP')
        #         conv = tf.nn.conv2d(self.h_pool, W, strides=[1, 1, 1, 1], padding='SAME', name='conv_GAP')
        #         print(conv)
        #         pool = tf.nn.avg_pool(conv, ksize=[1, 12, 1, 2], strides=[1, 12, 1, 2], padding='SAME')
        #         print(pool)
        #         self.scores_o = tf.reduce_mean(pool, axis=[1, 2])
        #         print(self.scores_o)
        #         self.predictions = tf.argmax(self.scores_o, 1, name="predictions")
        # else:
        #     self.h_pool_flat = tf.contrib.layers.flatten(self.h_pool)
        #     print(self.h_pool_flat)
        #
        #     # Add dropout
        #     with tf.name_scope("dropout1"):
        #         self.h_drop_1 = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
        #         print(self.h_drop_1)
        #
        #     with tf.name_scope("fc1"):
        #         W_fc1 = tf.get_variable("W_fc1", shape=[1536, 128], initializer=tf.contrib.layers.xavier_initializer())
        #         # W_fc1 = tf.get_variable("W_fc1", shape=[3328, 128], initializer=tf.contrib.layers.xavier_initializer())
        #         # W_fc1 = tf.get_variable("W_fc1", shape=[6400, 128], initializer=tf.contrib.layers.xavier_initializer())
        #         b_fc1 = tf.Variable(tf.constant(0.1, shape=[128]), name="b_fc1")
        #         # self.l2_loss_fc1 += tf.nn.l2_loss(W_fc1)
        #         # self.l2_loss_fc1 += tf.nn.l2_loss(b_fc1)
        #         self.z_fc1 = tf.nn.xw_plus_b(self.h_drop_1, W_fc1, b_fc1, name="scores_fc1")
        #         self.o_fc1 = tf.nn.relu(self.z_fc1, name="relu_fc1")
        #
        #     # Add dropout
        #     with tf.name_scope("dropout2"):
        #         self.h_drop_2 = tf.nn.dropout(self.o_fc1, self.dropout_keep_prob)
        #         print(self.h_drop_2)
        #
        #     with tf.name_scope("fc2"):
        #         W_fc2 = tf.get_variable("W_fc2", shape=[128, 64], initializer=tf.contrib.layers.xavier_initializer())
        #         b_fc2 = tf.Variable(tf.constant(0.1, shape=[64]), name="b_fc2")
        #         self.z_fc2 = tf.nn.xw_plus_b(self.h_drop_2, W_fc2, b_fc2, name="scores_fc2")
        #         self.o_fc2 = tf.nn.relu(self.z_fc2, name="relu_fc2")
        #
        #     # Add dropout
        #     with tf.name_scope("dropout3"):
        #         self.h_drop_3 = tf.nn.dropout(self.o_fc2, self.dropout_keep_prob)
        #         print(self.h_drop_3)
        #
        #     # Final (unnormalized) scores and predictions
        #     with tf.name_scope("output"):
        #         W_o = tf.get_variable("W_o", shape=[64, self.num_classes], initializer=tf.contrib.layers.xavier_initializer())
        #         b_o = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b_o")
        #         l2_loss += tf.nn.l2_loss(W_o)
        #         l2_loss += tf.nn.l2_loss(b_o)
        #         # self.scores_o = tf.reshape(self.h_drop_2, [-1, 128])
        #         self.scores_o = tf.nn.xw_plus_b(self.h_drop_3, W_o, b_o, name="scores_o")
        #         self.predictions = tf.argmax(self.scores_o, 1, name="predictions")
        #         print(self.scores_o)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            # losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores_o, labels=self.input_y)
            # self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss
            # self.loss = tf.reduce_mean(losses)

            self.loss = tf.losses.log_loss(labels=self.input_y, predictions=self.scores_o, loss_collection=tf.GraphKeys.LOSSES)
            # self.loss = tf.losses.get_losses(scope=None, loss_collection=tf.GraphKeys.LOSSES)
            # self.loss = tf.contrib.losses.log_loss(labels=self.input_y, predictions=self.scores_o)
            # self.loss = tf.contrib.losses.add_loss(
            #     tf.contrib.losses.log_loss(self.input_y, self.scores_o),
            #     loss_collection=tf.GraphKeys.LOSSES
            # )

    def set_placeholder(self):
        # Placeholders for input, output and dropout
        self.input_left = tf.placeholder(tf.int32, [None, self.sequence_length_left], name="input_left")
        self.input_right = tf.placeholder(tf.int32, [None, self.sequence_length_right], name="input_right")
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    def pad_for_wide_conv(self, x, w):
        return tf.pad(x, np.array([[0, 0], [w - 1, w - 1], [0, 0], [0, 0]]), "CONSTANT", name="pad_wide_conv")

    def make_attention_mat(self, x1, x2):
        # [batch, s, d, 1]
        # x1, x2 = [batch, height, width, 1] = [batch, d, s, 1]
        # x2 => [batch, height, 1, width]
        # [batch, width, wdith] = [batch, s, s]
        euclidean = tf.sqrt(tf.reduce_sum(tf.square(x1 - tf.matrix_transpose(x2)), axis=1))
        return 1 / (1 + euclidean)

    def cos_sim(self, v1, v2):
        v1_normed = tf.nn.l2_normalize(v1, dim=2, name=None)
        v2_normed = tf.nn.l2_normalize(v2, dim=2, name=None)

        dot_products = tf.reduce_mean(tf.transpose(tf.matmul(tf.transpose(v1_normed, perm=[0, 3, 1, 2]),
                                                             tf.transpose(v2_normed, perm=[0, 3, 2, 1])),
                                                   perm=[0, 2, 3, 1]), axis=3)
        print("cos_sim")
        print(v1_normed)
        print(dot_products)
        return dot_products

    def euclidean_score(self, v1, v2):
        euclidean = tf.sqrt(tf.reduce_sum(tf.square(v1 - v2), axis=1))
        return 1 / (1 + euclidean)

    def w_pool_att(self, x, attention, w, variable_scope):
        # 'abcnn2_pool_' + name
        # x: [batch, di, s+w-1, 1]
        # attention: [batch, s+w-1]
        with tf.variable_scope(variable_scope):
            print("col_wise_sum")
            pools = []
            # [batch, s+w-1] => [batch, 1, s+w-1, 1]
            print(attention)
            col_wise_sum = tf.reduce_sum(attention, axis=2)
            print(col_wise_sum)
            attention = tf.expand_dims(tf.expand_dims(col_wise_sum, -1), -1)
            print(attention)

            for i in range(FLAGS.sentence_len):
                pools.append(tf.reduce_sum(x[:, i:i+w, :, :] * attention[:, i:i+w, :, :], axis=1, keep_dims=True))

            w_ap = tf.concat(pools, axis=1, name="w_ap")
            print(w_ap)
            w_ap = tf.cast(w_ap, tf.float32)
            print(w_ap)

        return w_ap

    def branch_am_cnn(self, embedded_chars_expanded_left, embedded_chars_expanded_right, channel, width, filter_size,
                      num_filters, conv_pad, pool_pad, name, left_len, right_len, abcnn1=False, abcnn2=False):
        # Apply ABCNN-1
        if abcnn1:
            with tf.name_scope('abcnn1_mat_'+name):
                aW_left = tf.get_variable(name='aW_'+name+'_left', shape=(left_len, width),
                                          initializer=tf.contrib.layers.xavier_initializer(),
                                          regularizer=tf.contrib.layers.l2_regularizer(
                                              scale=self.l2_reg_lambda))  # [batch, s, s]
                aW_right = tf.get_variable(name='aW_'+name+'_right', shape=(right_len, width),
                                           initializer=tf.contrib.layers.xavier_initializer(),
                                           regularizer=tf.contrib.layers.l2_regularizer(
                                               scale=self.l2_reg_lambda))  # [batch, s, s]
                att_mat = self.cos_sim(embedded_chars_expanded_left, embedded_chars_expanded_right)  # [batch, s, s]

                print("ijk,kl->ijl")
                print(att_mat)
                print(aW_left)
                x1_a = tf.expand_dims(tf.einsum("ijk,kl->ijl", att_mat, aW_left), -1)
                x2_a = tf.expand_dims(tf.einsum("ijk,kl->ijl", tf.matrix_transpose(att_mat), aW_right), -1)

                embedded_chars_expanded_left = tf.concat([embedded_chars_expanded_left, x1_a], axis=3)
                embedded_chars_expanded_right = tf.concat([embedded_chars_expanded_right, x2_a], axis=3)

        with tf.name_scope("conv-maxpool-"+name+'_left'):
            # Convolution Layer
            filter_shape = [filter_size, width, channel, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W_'+name+'_left')
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b_'+name+'_left')
            embedded_chars_expanded = self.pad_for_wide_conv(embedded_chars_expanded_left, filter_size)
            conv = tf.nn.conv2d(embedded_chars_expanded, W, strides=[1, 1, width, 1], padding=conv_pad, name='conv_'+name+'_left')

            # Apply nonlinearity
            h_left = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu_'+name+'_left')
            h_left = tf.transpose(h_left, perm=[0, 1, 3, 2])

        with tf.name_scope("conv-maxpool-"+name+'_right'):
            # Convolution Layer
            filter_shape = [filter_size, width, channel, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W_'+name+'_right')
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b_'+name+'_right')
            embedded_chars_expanded = self.pad_for_wide_conv(embedded_chars_expanded_right, filter_size)
            conv = tf.nn.conv2d(embedded_chars_expanded, W, strides=[1, 1, width, 1], padding=conv_pad, name='conv_'+name+'_right')

            # Apply nonlinearity
            h_right = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu_'+name+'_right')
            h_right = tf.transpose(h_right, perm=[0, 1, 3, 2])
            print(h_right)

        # Apply ABCNN-2
        if abcnn2:
            with tf.name_scope('abcnn2_mat_' + name):
                att_mat = self.cos_sim(h_left, h_right)  # [batch, s, s]

                pooled_left = self.w_pool_att(h_left, att_mat, w=filter_size, variable_scope='abcnn2_pool_'+name+'_left')
                pooled_right = self.w_pool_att(h_right, tf.transpose(att_mat, [0, 2, 1]), w=filter_size, variable_scope='abcnn2_pool_'+name+'_right')
        else:
            # Maxpooling over the outputs
            pooled_left = tf.nn.avg_pool(h_left, ksize=[1, filter_size, 1, 1], strides=[1, 1, 1, 1], padding=pool_pad, name='pool_'+name+'_left')
            pooled_right = tf.nn.avg_pool(h_right, ksize=[1, filter_size, 1, 1], strides=[1, 1, 1, 1], padding=pool_pad, name='pool_'+name+'_right')
        print(pooled_left)
        print(pooled_right)
        return pooled_left, pooled_right


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, l2_reg_lambda=0.0):
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.l2_reg_lambda = l2_reg_lambda

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_right")
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        l2_loss = tf.constant(0.0)

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0), name="W_emb")
            print(self.W)
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            print(self.embedded_chars_expanded)

        h_conv1, pooled_2, pooled_3 = self.branch_am_cnn(self.embedded_chars_expanded)

        self.h_pool_flat = tf.contrib.layers.flatten(pooled_3)
        print(self.h_pool_flat)

        # self.h_pool_flat_1 = tf.contrib.layers.flatten(h_conv1)
        # print(self.h_pool_flat_1)
        # self.h_pool_flat_2 = tf.contrib.layers.flatten(pooled_2)
        # print(self.h_pool_flat_2)
        # self.h_pool_flat_3 = tf.contrib.layers.flatten(pooled_3)
        # print(self.h_pool_flat_3)
        # self.h_pool_flat = tf.concat([self.h_pool_flat_1, self.h_pool_flat_2, self.h_pool_flat_3], axis=1)
        # print(self.h_pool_flat)
        # pool_output = tf.reduce_mean(pool_output, axis=[1, 3])
        # print(pool_output)

        # Add dropout
        with tf.name_scope("dropout1"):
            self.h_drop_1 = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
            print(self.h_drop_1)

        with tf.name_scope("fc1"):
            W_fc1 = tf.get_variable("W_fc1", shape=[896, 128], initializer=tf.contrib.layers.xavier_initializer())
            b_fc1 = tf.Variable(tf.constant(0.1, shape=[128]), name="b_fc1")
            # self.l2_loss_fc1 += tf.nn.l2_loss(W_fc1)
            # self.l2_loss_fc1 += tf.nn.l2_loss(b_fc1)
            self.z_fc1 = tf.nn.xw_plus_b(self.h_drop_1, W_fc1, b_fc1, name="scores_fc1")
            self.o_fc1 = tf.nn.relu(self.z_fc1, name="relu_fc1")

        # Add dropout
        with tf.name_scope("dropout2"):
            self.h_drop_2 = tf.nn.dropout(self.o_fc1, self.dropout_keep_prob)
            print(self.h_drop_2)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W_o = tf.get_variable("W_o", shape=[128, self.num_classes], initializer=tf.contrib.layers.xavier_initializer())
            b_o = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b_o")
            l2_loss += tf.nn.l2_loss(W_o)
            l2_loss += tf.nn.l2_loss(b_o)
            # self.scores_o = tf.reshape(self.h_drop_2, [-1, 128])
            self.scores_o = tf.nn.xw_plus_b(self.h_drop_2, W_o, b_o, name="scores_o")
            self.predictions = tf.argmax(self.scores_o, 1, name="predictions")
            print(self.scores_o)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores_o, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss

    def branch_am_cnn(self, embedded_chars_expanded):
        filter_size_1, filter_size_2, filter_size_3 = 3, 3, 3
        num_filters_1, num_filters_2, num_filters_3 = 64, 64, 128
        with tf.name_scope("conv-maxpool-%s" % filter_size_1):
            # Convolution Layer
            filter_shape = [filter_size_1, self.embedding_size, 1, num_filters_1]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters_1]), name="b")
            conv = tf.nn.conv2d(embedded_chars_expanded, W, strides=[1, 1, self.embedding_size, 1], padding="SAME", name="conv1")
            # Apply nonlinearity
            h_conv1 = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu1")

            # Maxpooling over the outputs
            # pooled = tf.nn.max_pool(h, ksize=[1, self.sequence_length_left - filter_size_1 + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
            # pooled_1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID', name="pool1")
            print(h_conv1)
            # print(pooled)
            # pooled_outputs.append(pooled)
            # Add dropout
            h_conv1 = tf.nn.dropout(h_conv1, self.dropout_keep_prob)

        with tf.name_scope("conv-maxpool-%s" % filter_size_2):
            # Convolution Layer
            # filter_shape = [filter_size_2, self.embedding_size, 1, num_filters_2]
            filter_shape = [filter_size_2, 1, num_filters_1, num_filters_2]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters_2]), name="b")
            conv = tf.nn.conv2d(h_conv1, W, strides=[1, 1, 1, 1], padding="SAME", name="conv2")
            # Apply nonlinearity
            h_conv2 = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu2")
            # Maxpooling over the outputs
            # pooled = tf.nn.max_pool(h, ksize=[1, self.sequence_length_left - filter_size_2 + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
            pooled_2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID', name="pool2")
            print(h_conv2)
            print(pooled_2)
            pooled_2 = tf.nn.dropout(pooled_2, self.dropout_keep_prob)

        with tf.name_scope("conv-maxpool-%s" % filter_size_3):
            # Convolution Layer
            # filter_shape = [filter_size_3, self.embedding_size, 1, num_filters_3]
            filter_shape = [filter_size_3, 1, num_filters_2, num_filters_3]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters_3]), name="b")
            conv = tf.nn.conv2d(pooled_2, W, strides=[1, 1, 1, 1], padding="SAME", name="conv3")
            # Apply nonlinearity
            h_conv3 = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu3")
            # Maxpooling over the outputs
            # pooled = tf.nn.max_pool(h, ksize=[1, self.sequence_length_left - filter_size_3 + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
            pooled_3 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID', name="pool3")
            print(h_conv3)
            print(pooled_3)

        return h_conv1, pooled_2, pooled_3


class Train:
    # def show_prediction(self):
    #     dev_batches = DataHelpers().batch_iter(list(zip(x_dev, y_dev)), FLAGS.batch_size, 1)
    #     total_dev_correct = 0
    #     total_dev_loss = 0
    #     print("\nEvaluation:")
    #     for dev_batch in dev_batches:
    #         x_dev_batch, y_dev_batch = zip(*dev_batch)
    #         loss, dev_correct = dev_step(x_dev_batch, y_dev_batch)
    #         total_dev_correct += dev_correct * len(y_dev_batch)

    def train(self, x_train, y_train, x_dev, y_dev, vocab_processor):
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, log_device_placement=FLAGS.log_device_placement)
            sess = tf.Session(config=session_conf)
            # sess = tf.Session()
            with sess.as_default():
                # cnn = TextCNN(sequence_length=x_train.shape[1],
                #     num_classes=FLAGS.sentiment_class,
                #     vocab_size=len(vocab_processor.vocabulary_),
                #     embedding_size=FLAGS.embedding_dim)
                cnn = Text_BiLSTM(sequence_length=x_train.shape[1],
                              num_classes=FLAGS.sentiment_class,
                              vocab_size=len(vocab_processor.vocabulary_),
                              embedding_size=FLAGS.embedding_dim)

                # Define Training procedure
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)
                grads_and_vars = optimizer.compute_gradients(cnn.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
                # train_op = tf.train.AdamOptimizer(learning_rate=FLAGS.lr, beta1=0.9, beta2=0.999,
                #                                         epsilon=1e-8).minimize(cnn.loss)

                # Keep track of gradient values and sparsity (optional)
                grad_summaries = []
                for g, v in grads_and_vars:
                    if g is not None:
                        grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                        sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                        grad_summaries.append(grad_hist_summary)
                        grad_summaries.append(sparsity_summary)
                grad_summaries_merged = tf.summary.merge(grad_summaries)

                # Output directory for models and summaries
                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
                print("Writing to {}\n".format(out_dir))

                # Summaries for loss and accuracy
                loss_summary = tf.summary.scalar("loss", cnn.loss)
                acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

                # Train Summaries
                train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
                train_summary_dir = os.path.join(out_dir, "summaries", "train")
                train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

                # Dev summaries
                dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
                dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
                dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

                # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

                # Write vocabulary
                vocab_processor.save(os.path.join(out_dir, "vocab"))

                # Initialize all variables
                sess.run(tf.global_variables_initializer())

                def train_step(x_batch, y_batch):
                    """
                    A single training step
                    """
                    feed_dict = {
                        cnn.input_x: x_batch,
                        cnn.input_y: y_batch,
                        cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                    }
                    _, step, summaries, loss, accuracy = sess.run(
                        [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                        feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    train_summary_writer.add_summary(summaries, step)

                def dev_step(x_batch, y_batch, writer=None):
                    """
                    Evaluates model on a dev set
                    """
                    feed_dict = {
                        cnn.input_x: x_batch,
                        cnn.input_y: y_batch,
                        cnn.dropout_keep_prob: 1.0
                    }
                    step, summaries, loss, accuracy = sess.run([global_step, dev_summary_op, cnn.loss, cnn.accuracy], feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    if writer:
                        writer.add_summary(summaries, step)
                    return loss, accuracy

                # Generate batches
                batches = DataHelpers().batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
                # Training loop. For each batch...
                for batch in batches:
                    x_batch, y_batch = zip(*batch)
                    train_step(x_batch, y_batch)
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % FLAGS.evaluate_every == 0:
                        dev_batches = DataHelpers().batch_iter(list(zip(x_dev, y_dev)), FLAGS.batch_size, 1)
                        total_dev_correct = 0
                        total_dev_loss = 0
                        print("\nEvaluation:")
                        for dev_batch in dev_batches:
                            x_dev_batch, y_dev_batch = zip(*dev_batch)
                            loss, dev_correct = dev_step(x_dev_batch, y_dev_batch)
                            total_dev_correct += dev_correct * len(y_dev_batch)
                            total_dev_loss += loss * len(y_dev_batch)
                            # dev_step(x_left_dev, x_right_dev, y_dev, writer=dev_summary_writer)
                        dev_accuracy = float(total_dev_correct) / len(y_dev)
                        dev_loss = float(total_dev_loss) / len(y_dev)
                        print('Accuracy on dev set: {0}, loss on dev set: {1}'.format(dev_accuracy, dev_loss))
                        print("Evaluation finished")
                    if current_step % FLAGS.checkpoint_every == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))

                # self.show_prediction()

    def preprocess(self):
        # 读取训练数据
        data = pd.read_csv(FLAGS.train_data_file, sep=",", error_bad_lines=False)

        print(pd.value_counts(data['subject']))
        print(pd.value_counts(data['sentiment_value']))
        print(pd.value_counts(data['sentiment_word']))

        # 根据sentiment word构建字典
        # sentiment_word = set(data['sentiment_word'])
        # sentiment_word.remove(np.nan)
        # with open(FLAGS.dictionary, 'w') as f:
        #     for word in sentiment_word:
        #         print(word)
        #         f.write(word+'\n')
        # f.close()
        # print("dictionary done!")

        data = data.fillna('空')

        # 数据切分
        data = DataHelpers().sentence_cut(data=data, dict=True)
        data[['sentence_seq']].to_csv('D:/Data/sentence/train.csv', encoding='utf8', index=False)

        # Build vocabulary
        # max_document_length = max([len(x.split(" ")) for x in x_text])
        max_document_length = FLAGS.sentence_len
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length, min_frequency=5)
        vocab_processor.fit(data['sentence_seq'])
        # x = np.array(list(vocab_processor.fit_transform(x_text)))
        x = np.array(list(vocab_processor.transform(data['sentence_seq'])))

        subject_dict = {'动力': 0, '价格': 1, '油耗': 2, '操控': 3, '舒适性': 4, '配置': 5, '安全性': 6, '内饰': 7, '外观': 8, '空间': 9}
        subject_numerical = []
        for subject in data['subject']:
            subject_numerical.append(subject_dict[subject])
        y = to_categorical(data['sentiment_value'], num_classes=FLAGS.sentiment_class)
        # y = to_categorical(subject_numerical, num_classes=FLAGS.subject_class)

        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = x[shuffle_indices]
        y_shuffled = y[shuffle_indices]

        dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
        x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
        y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
        del x, y, x_shuffled, y_shuffled

        return x_train, y_train, x_dev, y_dev, vocab_processor


if __name__ == '__main__':
    # 模型训练
    obj_train = Train()
    x_train, y_train, x_dev, y_dev, vocab_processor = obj_train.preprocess()
    obj_train.train(x_train, y_train, x_dev, y_dev, vocab_processor)
