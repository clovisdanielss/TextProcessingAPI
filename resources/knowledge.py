from resources.clustering import to_embed_vector
from util import send_message
import json
import flask_restful
import flask
from numpy.linalg import norm
import gc
from initialize import jobs, api
from uuid import uuid4
from threading import Thread
from time import time


def intent_threshold(data, output_mean=None):
    output = to_embed_vector(data).numpy()
    if output_mean is None:
        output_mean = output.mean(axis=0)
    return avg_distance(output, output_mean)


def avg_distance(data, data_mean):
    dist = 0
    for x in data:
        dist += norm(x - data_mean)
    return dist / data.shape[0]

def euclidian_mean(data, intent_reference, uuid):
    threshold = intent_threshold(intent_reference)
    clusters = {}
    for entry in data["clusters"]:
        if entry["cluster"] not in clusters.keys():
            clusters[entry["cluster"]] = [entry["phrase"]]
        else:
            clusters[entry["cluster"]] += [entry["phrase"]]
    jobs[uuid] = [{"cluster": key,
             "threshold": (value := intent_threshold(clusters[key], output_mean=data["cluster_centers"][key]) if len(clusters[key]) > 1 else 999),
             "intent_test": "CANDIDATE" if value < threshold else "IGNORED"} for key in clusters.keys()]

def nearest_clusters(data, intent_reference, uuid):
    center = to_embed_vector(intent_reference).numpy()
    center = center.mean(axis=0)
    threshold = intent_threshold(intent_reference, center)
    api.app.logger.info("Job %s - Threshold %s", uuid, threshold)
    clusters = {}
    for entry in data["clusters"]:
        if entry["cluster"] not in clusters.keys():
            clusters[entry["cluster"]] = [entry["phrase"]]
        else:
            clusters[entry["cluster"]] += [entry["phrase"]]
    jobs[uuid] = [{"cluster": key,
             "threshold": (value := norm(data["cluster_centers"][key] - center) if len(clusters[key]) > 1 else 999),
             "intent_test": "CANDIDATE" if norm(data["cluster_centers"][key] - center) <= threshold else "IGNORED"} for key in clusters.keys()]

def check_clusters(data, intent_reference, algorithm, uuid):
    api.app.logger.info("Job %s - Algorithm %s", uuid, algorithm)
    time_start = time()
    if algorithm == "euclidian_mean":
        euclidian_mean(data, intent_reference, uuid)
    elif algorithm == "neareast_clusters":
        nearest_clusters(data, intent_reference, uuid)
    time_total = time() - time_start
    api.app.logger.info("Job %s - Done in %s", uuid, str(time_total))



class Knowledge(flask_restful.Resource):
    
    def get(self, job_id):
        if job_id in jobs.keys():
            result = jobs[job_id]
            del jobs[job_id]
            gc.collect()
            return send_message(result, label="data"), 200
        return send_message("Job not found"), 404
        
        
    def post(self):
        try:
            body = flask.request.json
            if "intent_reference" not in body.keys():
                raise Exception("intent_reference from knowledge intent must not be None")
            if "data" not in body.keys():
                raise Exception("data to analysis must not be None")
            algorithm = flask.request.args.get('algorithm')
            if algorithm is None:
                algorithm = "euclidian_mean"
            uuid = str(uuid4())
            thread = Thread(target=check_clusters, args=(body["data"], body["intent_reference"], algorithm, uuid))
            thread.start()
            return send_message(uuid, label="id"), 200
        except Exception as e:
            return send_message(str(e), label="error"), 500
