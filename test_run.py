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

#x_raw = ["Mohamed was gay for porky pig", "saaale madarchord cheats!! indian Muslims are shameless!! refund their money immediately refund, reimburse every expense.!!  if you are not ready to marry your daughter to a far off land why did you consent? you should have discussed it with your relatives!! <br/>I am a person who can't tolerate injustice. yes they may be Afghans of Pakistanis. don't do any injustice to anyone. this is like inviting someone to get insulted. this is not Indian ethos!! <br/><br/>indian ethos speaks once you commit something to someone come what may stick to your words. this is what Ramayan teaches us!!  even Ravan knowing ram is going to kill him in war he came down to perform particular puja for ram. <br/>this treacherous family should be thrown out from the country. they have shamed India."]

text = input("enter a comment : ")
x_raw = [text]
#y_test = [1, 1]
y_test = None

# Map data into vocabulary

checkpoint_dir = "/opt/workarea/TextCNN/cnn-text-classification-tf/runs/1542213092/checkpoints"

#vocab_path = checkpoint_dir + "/.." + "/vocab"
vocab_path = '/opt/spamtest_yash/vocab'

vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

with tf.Session() as sess:
    #model_filename ='/opt/workarea/TextCNN/cnn-text-classification-tf/runs/1542213092/checkpoints/frozen_model.pb'
    #model_filename = '/opt/workarea/TextCNN/cnn-text-classification-tf/yash_test/frozen_model_y_1.pb'
    model_filename = '/opt/spamtest_yash/frozen_model_y_1.pb'

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
out_path = '/opt/spamtest_yash/prediction.csv'

print("Saving evaluation to {0}".format(out_path))
print(predictions_human_readable)
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)

