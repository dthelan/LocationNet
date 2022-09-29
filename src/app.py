import tensorflow as tf
from tensorflow import keras

from flask import Flask, request, jsonify

import numpy as np
import geohash
import json
import heapq
import io

# Turn down warning level
tf.get_logger().setLevel('WARN')

app = Flask(__name__)
app.config['DEBUG'] = True

with open('../data/char_dict.json', 'r') as f:
    char_dict = json.load(f)

chars = {v: k for k, v in char_dict.items()}


def generate_geohash(points):
    point_list = [geohash.encode(p[0], p[1], precision=6) for p in points]
    points_flat = [i for k in point_list for i in k]
    return points_flat


def build_vector(input_sq, char_idx):
    x = np.zeros((1, 23, len(char_idx)), dtype=bool)
    for t, char in enumerate(input_sq):
        x[0, t, char_idx[char]] = 1
    return x


def predict_from_vec(model, x, char_map, position):
    with tf.device('/cpu:0'):
        results = model.predict(x, batch_size=1, verbose=0)[0]

    next_index = argument(results, position)
    next_char = char_map[next_index]
    return next_char, float(results[next_index])


def argument(results, position):
    top_3 = heapq.nlargest(3, results)
    print(top_3)
    prediction_index = [list(results).index(i) for i in top_3][position]
    return prediction_index


def predict(model, points_flat, char_idx):
    locations = {'p1': {}, 'p2': {}, 'p3': {}, 'p4': {}, 'p5': {}, 'p6': {}}
    n = 0
    # seed = [q for p in [p1, p2, p3] for q in p]
    while n < 6:
        if n == 0:
            x = build_vector(points_flat, char_idx)
            next_char, prob_current = predict_from_vec(model, x, chars, 0)
            locations['p' + str(n + 1)].update({next_char: prob_current})
            if prob_current < 0.01:
                next_char, prob_current = predict_from_vec(model, x, chars, 1)
                locations['p' + str(n + 1)].update({next_char: prob_current})
                next_char, prob_current = predict_from_vec(model, x, chars, 2)
                locations['p' + str(n + 1)].update({next_char: prob_current})
        if n > 0:
            for i in locations['p' + str(n)]:
                current_seed = points_flat + [k for k in i]
                x = build_vector(current_seed, char_idx)
                next_char, prob_current = predict_from_vec(model, x, chars, 0)
                string = i + next_char
                locations['p' + str(n + 1)].update({string: prob_current * locations['p' + str(n)][i]})
                if prob_current < 0.01:
                    next_char, prob_current = predict_from_vec(model, x, chars, 1)
                    string = i + next_char
                    locations['p' + str(n + 1)].update({string: prob_current * locations['p' + str(n)][i]})
                    next_char, prob_current = predict_from_vec(model, x, chars, 2)
                    string = i + next_char
                    locations['p' + str(n + 1)].update({string: prob_current * locations['p' + str(n)][i]})
        n += 1
    return locations


@app.route('/')
def index():
    """ Displays the index page accessible at '/'
    """
    return "Ping"


@app.route('/predict', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        input_dict = json.load(io.BytesIO(request.data))

        points = list(input_dict.values())

        points_flat = generate_geohash(points)

        predictions = predict(network, points_flat, char_dict)

        print(predictions)
        return jsonify(**predictions)


if __name__ == "__main__":
    with tf.device('/cpu:0'):
        network = keras.models.load_model('../models/example_model')

    app.run(host='0.0.0.0', port=8001)
