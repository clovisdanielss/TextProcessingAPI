from resources.clustering import to_embed_vector
from util import send_message
import json
import flask_restful
import flask
from numpy.linalg import norm


def intent_threshold(data):
    output = to_embed_vector(data).numpy()
    output_mean = output.mean(axis=0)
    return avg_distance(output, output_mean)


def avg_distance(data, data_mean):
    dist = 0
    for x in data:
        dist += norm(x - data_mean)
    return dist / data.shape[0]

def check_clusters(data, intent_reference):
    threshold = intent_threshold(intent_reference)
    clusters = {}
    for entry in data:
        if entry["cluster"] not in clusters.keys():
            clusters[entry["cluster"]] = [entry["phrase"]]
        else:
            clusters[entry["cluster"]] += [entry["phrase"]]
    return str([(key, value) for key in clusters.keys() if (value := intent_threshold(clusters[key])) < threshold])



class Knowledge(flask_restful.Resource):
    def post(self):
        try:
            body = flask.request.json
            if "intent_reference" not in body.keys():
                raise Exception("intent_reference from knowledge intent must not be None")
            if "data" not in body.keys():
                raise Exception("data to analysis must not be None")
            return send_message(check_clusters(body["data"], body["intent_reference"]), label="data"), 200
        except Exception as e:
            return send_message(str(e), label="error"), 500
