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
from flask import Flask, request
from flask_cors import CORS
import sys
from flask import jsonify
import json

if sys.version_info[0] < 3:
    raise Exception("Python 3 or a more recent version is required.")

##################################################
# API part
##################################################
app = Flask(__name__)
cors = CORS(app)

vocab_processor = None
persistent_sess = None
input_x = None
dropout_keep_prob = None
predictions = None


def setup():
    global vocab_processor
    global persistent_sess
    global input_x
    global dropout_keep_prob
    global predictions
    
    # base_dir = "/Users/yash.dalmia/spamWork/spamtest_yash/model_ori/"
    base_dir = "/opt/spamtest_yash/deploy_py/model_ori/"
    vocab_path = base_dir + 'vocab'
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    print("\nEvaluating...\n")

    model_filename = base_dir + 'frozen_model.pb'

    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)
        # Get the placeholders from the graph by name
        input_x = tf.get_default_graph().get_operation_by_name("import/input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = tf.get_default_graph().get_operation_by_name("import/dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = tf.get_default_graph().get_operation_by_name("import/output/predictions").outputs[0]
        persistent_sess = tf.Session(graph=g_in)


@app.route("/")
def health_check():
    return "Working OK !"


@app.route("/test_predict", methods=['GET'])
def test_predict():
    x_in = ['alskdfjalsdf', 'gaumutras women']
    x_test = np.array(list(vocab_processor.transform(x_in)))

    # Generate batches for one epoch
    batches = data_helpers.batch_iter(list(x_test), 1, 1, shuffle=False)

    # Collect the predictions here
    all_predictions = []

    for x_test_batch in batches:
        batch_predictions = persistent_sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
        all_predictions = np.concatenate([all_predictions, batch_predictions])
    print(all_predictions)
    return str(all_predictions)


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        req_content = request.get_json()
        input_comment_text = req_content['cText']
    else:
        input_comment_text = request.args.get('cText')
    print('input comment : ' + input_comment_text)

    x_in = [input_comment_text]
    x_test = np.array(list(vocab_processor.transform(x_in)))

    # Generate batches for one epoch
    batches = data_helpers.batch_iter(list(x_test), 1, 1, shuffle=False)

    # Collect the predictions here
    all_predictions = []

    for x_test_batch in batches:
        batch_predictions = persistent_sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
        all_predictions = np.concatenate([all_predictions, batch_predictions])
    if str(all_predictions[0]) == '1.0':
        is_spam = True
    else:
        is_spam = False
    return_dict = {"cText": input_comment_text, "isSpam": is_spam}
    # return_json = json.dumps(return_dict)
    return jsonify(return_dict)


setup()

##################################################
# END API part
##################################################


if __name__ == "__main__":
    app.run(debug=True)

