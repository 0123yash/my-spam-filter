#! /usr/bin/env python

import sys
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
from tensorflow.python.platform import gfile
import helper_funcs

csvFileName = 'all_comments_19Nov.csv'
csvFilePath = '/opt/spamtest_yash/data_to_run/' + csvFileName
commentsListFromCsv = helper_funcs.getListFromCsv(csvFilePath)

x_raw = [x[1] for x in commentsListFromCsv if x]
#x_raw = ["gaumutra women", "asdfasdfas"]

#print (x_raw[0], x_raw[0])
#y_test = [1, 1]
y_test = None

# Map data into vocabulary

base_dir = "/opt/spamtest_yash/model_ori/"
#checkpoint_dir = "/opt/workarea/TextCNN/cnn-text-classification-tf/runs/1542213092/checkpoints"

#vocab_path = checkpoint_dir + "/.." + "/vocab"
#vocab_path = '/opt/spamtest_yash/vocab'
vocab_path = base_dir + 'vocab'

vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

with tf.Session() as sess:
    #model_filename ='/opt/workarea/TextCNN/cnn-text-classification-tf/runs/1542213092/checkpoints/frozen_model.pb'
    #model_filename = '/opt/workarea/TextCNN/cnn-text-classification-tf/yash_test/frozen_model_y_1.pb'
    model_filename = base_dir + 'frozen_model.pb'

    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)
        # output_node_names =[n.name for n in tf.get_default_graph().as_graph_def().node]
        # print (output_node_names)
        # Get the placeholders from the graph by name
        input_x = tf.get_default_graph().get_operation_by_name("import/input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = tf.get_default_graph().get_operation_by_name("import/dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = tf.get_default_graph().get_operation_by_name("import/output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), 1, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

print (all_predictions)

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
# out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
# out_path = checkpoint_dir + "/.." + "/prediction.csv"
#out_path = "/opt/workarea/TextCNN/cnn-text-classification-tf/yash_test/prediction.csv"
out_file_name = csvFileName + '_prediction.csv'
out_path = base_dir + out_file_name

print("Saving evaluation to {0}".format(out_path))
#print(predictions_human_readable)
with open(out_path, 'w', encoding='utf-8') as f:
    csv.writer(f).writerows(predictions_human_readable)

