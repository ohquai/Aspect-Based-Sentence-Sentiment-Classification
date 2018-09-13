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
tf.flags.DEFINE_string("pretrained_word_emb", "./utils/word2vec.txt", "stoplist path.")
tf.flags.DEFINE_string("model_data_path", "D:/DF/sentence_theme_based_sentiment/model/", "model path for storing.")

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("subject_class", 10, "number of classes (default: 2)")
tf.flags.DEFINE_integer("sentiment_class", 3, "number of classes (default: 2)")
tf.flags.DEFINE_integer("subject_sentiment_class", 30, "number of classes (default: 2)")
tf.flags.DEFINE_float("lr", 0.002, "learning rate (default: 0.002)")
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("sentence_len", 30, "Maximum length for sentence pair (default: 50)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.3, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.2, "L2 regularization lambda (default: 0.0)")

# LSTM Hyperparameters
tf.flags.DEFINE_integer("hidden_dim", 128, "Number of filters per filter size (default: 128)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 300, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
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
        stop_words = [' ', '我', '你', '还', '会', '因为', '所以', '这', '是', '和', '他们',
                      '了', '的', '也', '哦', '这个', '啊', '说', '知道', '哪里', '吧', '哪家',
                      '想', '啥', '怎么', '呢', '那', '嘛', '么',
                      '有', '指', '楼主', '私信', '谁', '可能', '像', '这样', '到底', '哪个', '看', '我们',
                      '只能', '主要', '些', '认为', '肯定',
                      ]

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


class Text_BiLSTM(object):
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
        self.embedding_placeholder = tf.placeholder(tf.float32, [self.vocab_size, self.embedding_size], name="pretrained_emb")

        # with tf.device('/cpu:0'), tf.name_scope("embedding"):
        #     self.W = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0), name="W_emb")
        #     print(self.W)
        #     self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
        #     self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
        #     print(self.embedded_chars_expanded)

        # h_conv1, pooled_2, pooled_3 = self.branch_am_cnn(self.embedded_chars_expanded)
        self.lookup_layer_op()
        self.biLSTM_layer_op()
        # self.scores_o = self.project_layer_op()
        # print(self.scores_o)
        # self.h_pool_flat = tf.contrib.layers.flatten(pooled_3)
        # print(self.h_pool_flat)
        #
        #
        # # Add dropout
        # with tf.name_scope("dropout1"):
        #     self.h_drop_1 = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
        #     print(self.h_drop_1)
        #
        # with tf.name_scope("fc1"):
        #     W_fc1 = tf.get_variable("W_fc1", shape=[896, 128], initializer=tf.contrib.layers.xavier_initializer())
        #     b_fc1 = tf.Variable(tf.constant(0.1, shape=[128]), name="b_fc1")
        #     # self.l2_loss_fc1 += tf.nn.l2_loss(W_fc1)
        #     # self.l2_loss_fc1 += tf.nn.l2_loss(b_fc1)
        #     self.z_fc1 = tf.nn.xw_plus_b(self.h_drop_1, W_fc1, b_fc1, name="scores_fc1")
        #     self.o_fc1 = tf.nn.relu(self.z_fc1, name="relu_fc1")
        #
        # # Add dropout
        # with tf.name_scope("dropout2"):
        #     self.h_drop_2 = tf.nn.dropout(self.o_fc1, self.dropout_keep_prob)
        #     print(self.h_drop_2)

        # Final (unnormalized) scores and predictions
        # with tf.name_scope("output"):
        #     # W_o = tf.get_variable("W_o", shape=[128, self.num_classes], initializer=tf.contrib.layers.xavier_initializer())
        #     # b_o = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b_o")
        #     # l2_loss += tf.nn.l2_loss(W_o)
        #     # l2_loss += tf.nn.l2_loss(b_o)
        #     # # self.scores_o = tf.reshape(self.h_drop_2, [-1, 128])
        #     # self.scores_o = tf.nn.xw_plus_b(self.h_drop_2, W_o, b_o, name="scores_o")
        #     self.predictions = tf.argmax(self.scores_o, 1, name="predictions")
        #     print(self.predictions)
        #
        # # Accuracy
        # with tf.name_scope("accuracy"):
        #     correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
        #     self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        #
        # # Calculate mean cross-entropy loss
        # with tf.name_scope("loss"):
        #     losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores_o, labels=self.input_y)
        #     self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss

    def biLSTM_layer_op(self):
        l2_loss = tf.constant(0.0)

        with tf.variable_scope("bi-lstm"):
            n_layers = 1
            x = tf.transpose(self.word_embeddings, [1, 0, 2])
            print('1111')
            print(x)
            # # Reshape to (n_steps*batch_size, n_input)
            x = tf.reshape(x, [-1, self.embedding_size])
            # # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
            # # x = tf.split(x, n_steps, 0)
            x = tf.split(axis=0, num_or_size_splits=self.sequence_length, value=x)
            print(x)
            # Define lstm cells with tensorflow
            # Forward direction cell
            with tf.name_scope("fw_biLSTM"), tf.variable_scope("fw_biLSTM"):
                print(tf.get_variable_scope().name)

                # fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                # lstm_fw_cell = rnn.DropoutWrapper(fw_cell, output_keep_prob=dropout)
                # lstm_fw_cell_m = rnn.MultiRNNCell([lstm_fw_cell]*n_layers, state_is_tuple=True)
                def lstm_fw_cell():
                    fw_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.hidden_dim, forget_bias=1.0, state_is_tuple=True)
                    return tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=self.dropout_keep_prob)

                # lstm_fw_cell_m = tf.contrib.rnn.MultiRNNCell([lstm_fw_cell() for _ in range(n_layers)], state_is_tuple=True)
                fw_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.hidden_dim, forget_bias=1.0, state_is_tuple=True)
                print(fw_cell)
                lstm_fw_cell_m = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=self.dropout_keep_prob)



                # Backward direction cell
            with tf.name_scope("bw_biLSTM"), tf.variable_scope("bw_biLSTM"):
                # bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                # lstm_bw_cell = rnn.DropoutWrapper(bw_cell, output_keep_prob=dropout)
                # lstm_bw_cell_m = rnn.MultiRNNCell([lstm_bw_cell]*n_layers, state_is_tuple=True)
                def lstm_bw_cell():
                    bw_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.hidden_dim, forget_bias=1.0, state_is_tuple=True)
                    return tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=self.dropout_keep_prob)

                # lstm_bw_cell_m = tf.contrib.rnn.MultiRNNCell([lstm_bw_cell() for _ in range(n_layers)], state_is_tuple=True)
                bw_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.hidden_dim, forget_bias=1.0, state_is_tuple=True)
                lstm_bw_cell_m = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=self.dropout_keep_prob)

            # Get lstm cell output
            # try:
            with tf.name_scope("full_biLSTM"), tf.variable_scope("full_biLSTM"):
                # outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x, dtype=tf.float32)
                # self.output, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x, dtype=tf.float32)
                output, state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_m, lstm_bw_cell_m, self.word_embeddings, dtype=tf.float32)
                # outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x, dtype=tf.float32)
                #         except Exception: # Old TensorFlow version only returns outputs not states
                #             outputs = tf.nn.bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x,
                #                                             dtype=tf.float32)
                print('2222')
                print(output)
                self.output = tf.concat(output, 2)
                print(self.output)
            # return outputs[-1]
            # return outputs

            with tf.name_scope("mean_pooling_layer"):
                self.out_put = tf.reduce_mean(self.output, 1)
                avg_pool = tf.nn.dropout(self.out_put, keep_prob=self.dropout_keep_prob)
                print("pool", avg_pool)

            with tf.name_scope('output'):
                # 双向
                W = tf.Variable(tf.truncated_normal([int(2*FLAGS.hidden_dim), self.num_classes], stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name='b')
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.logits = tf.nn.xw_plus_b(avg_pool, W, b, name='scores')
                self.y_pred_cls = tf.argmax(self.logits, 1, name='predictions')

            with tf.name_scope("loss"):
                # 损失函数，交叉熵
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
                self.loss = tf.reduce_mean(cross_entropy)+self.l2_reg_lambda * l2_loss

            with tf.name_scope("accuracy"):
                # 准确率
                correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
                self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def project_layer_op(self):
        with tf.variable_scope("proj"):
            W = tf.get_variable(name="W",
                                shape=[2 * FLAGS.hidden_dim, self.num_classes],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[self.num_classes],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

            s = tf.shape(self.output)
            #此时output的shape{batch_size*sentence,2*hidden_dim]
            self.output = tf.reshape(self.output, [-1, 2*FLAGS.hidden_dim])
            #pred的shape为[batch_size*sentence,num_classes]
            pred = tf.matmul(self.output, W) + b

            # pred = tf.nn.tanh(pred, name='tanh_layer')  # CT
            #logits的shape为[batch,sentence,num_classes]
            self.logits = tf.reshape(pred, [-1, s[1], self.num_classes])
            print(self.logits)
            return self.logits

    def lookup_layer_op(self):
        with tf.variable_scope("words"):

            # self._word_embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0), dtype=tf.float32, trainable=True, name="W_emb")
            # word_embeddings = tf.nn.embedding_lookup(params=self._word_embeddings, ids=self.input_x, name="word_embeddings")

            W = tf.Variable(tf.constant(0.0, shape=[self.vocab_size, self.embedding_size]), trainable=False, name="W")
            embedding_init = W.assign(self.embedding_placeholder)
            word_embeddings = tf.nn.embedding_lookup(params=W, ids=self.input_x, name="word_embeddings")

        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout_keep_prob)


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
    def load_word2vec(self, filename):
        vocab = []
        embd = []
        file = open(filename, 'r', encoding='utf8')
        for line in file.readlines():
            row = line.strip().split(' ')
            vocab.append(row[0])
            embd.append(row[1:])
        print('Loaded GloVe!')
        file.close()
        return vocab, embd

    def train(self, x_train, y_train, x_dev, y_dev, vocab_processor, embedding):
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
                              num_classes=FLAGS.subject_sentiment_class,
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
                        cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
                        cnn.embedding_placeholder: embedding
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
                        cnn.dropout_keep_prob: 1.0,
                        cnn.embedding_placeholder: embedding
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

        vocab, embd = self.load_word2vec(FLAGS.pretrained_word_emb)
        vocab_size = len(vocab)
        embedding_dim = len(embd[0])
        embedding = np.asarray(embd)

        # Build vocabulary
        # max_document_length = max([len(x.split(" ")) for x in x_text])
        max_document_length = FLAGS.sentence_len
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length, min_frequency=2)
        # vocab_processor.fit(data['sentence_seq'])
        vocab_processor.fit(vocab)
        # x = np.array(list(vocab_processor.fit_transform(x_text)))
        x = np.array(list(vocab_processor.transform(data['sentence_seq'])))

        # subject_dict = {'动力': 0, '价格': 1, '油耗': 2, '操控': 3, '舒适性': 4, '配置': 5, '安全性': 6, '内饰': 7, '外观': 8, '空间': 9}
        # subject_numerical = []
        # for subject in data['subject']:
        #     subject_numerical.append(subject_dict[subject])
        # y = to_categorical(data['sentiment_value'], num_classes=FLAGS.sentiment_class)
        # y = to_categorical(subject_numerical, num_classes=FLAGS.subject_class)

        subject_dict = {'动力_-1': 0, '价格_-1': 1, '油耗_-1': 2, '操控_-1': 3, '舒适性_-1': 4, '配置_-1': 5, '安全性_-1': 6, '内饰_-1': 7, '外观_-1': 8, '空间_-1': 9,
                        '动力_0': 10, '价格_0': 11, '油耗_0': 12, '操控_0': 13, '舒适性_0': 14, '配置_0': 15, '安全性_0': 16, '内饰_0': 17, '外观_0': 18, '空间_0': 19,
                        '动力_1': 20, '价格_1': 21, '油耗_1': 22, '操控_1': 23, '舒适性_1': 24, '配置_1': 25, '安全性_1': 26, '内饰_1': 27, '外观_1': 28, '空间_1': 29}
        data['subject_senti'] = data['subject']+'_'+data['sentiment_value'].astype('str')
        subject_numerical = []
        for subject in data['subject_senti']:
            print(subject)
            subject_numerical.append(subject_dict[subject])
        y = to_categorical(subject_numerical, num_classes=FLAGS.subject_sentiment_class)

        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = x[shuffle_indices]
        y_shuffled = y[shuffle_indices]

        dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
        x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
        y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
        del x, y, x_shuffled, y_shuffled

        return x_train, y_train, x_dev, y_dev, vocab_processor, embedding


if __name__ == '__main__':
    # 模型训练
    obj_train = Train()
    x_train, y_train, x_dev, y_dev, vocab_processor, embedding = obj_train.preprocess()
    obj_train.train(x_train, y_train, x_dev, y_dev, vocab_processor, embedding)
