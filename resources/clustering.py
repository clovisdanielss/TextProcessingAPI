from util import send_message
import uuid
from threading import Thread
import flask
import flask_restful
import numpy as np
import tensorflow_text
import nltk
from sklearn.cluster import KMeans
import tensorflow as tf
import gap_statistic
from initialize import jobs, sem, embed


def to_embed_vector(data, batch=10, i=0, size=None):
    if size is None or size > len(data):
        size = len(data)
    output = None
    while i < size:
        if i == 0:
            output = embed(data[i:i + batch])
        else:
            if i + batch < size:
                output = tf.concat([output, embed(data[i:i + batch])], 0)
            else:
                output = tf.concat([output, embed(data[i:size])], 0)
        i += batch
    return output


def most_frequent_words(data, class_sample, cluster, n=3):
    phrases = [data[i] for i in range(len(cluster.labels_)) if cluster.labels_[i] == class_sample]
    frequency = {}
    for phrase in phrases:
        tokens = nltk.word_tokenize(phrase)
        for token in tokens:
            token = token.lower()
            if token in nltk.corpus.stopwords.words("portuguese"):
                continue
            if not token.isalpha():
                continue
            if token in frequency.keys():
                frequency[token] += 1
            else:
                frequency[token] = 1
    frequency = sorted(frequency.items(), key=lambda item: item[1], reverse=True)
    return [word for word, count in frequency[:n]]


def nearest_words(data, output, class_sample, cluster, n=5):
    words = most_frequent_words(data, class_sample, cluster, n=n)
    indices = [i for i in range(len(cluster.labels_)) if cluster.labels_[i] == class_sample]
    dist = np.inf
    nearest_word = None
    for word in words:
        mean = 0
        for i in indices:
            mean += np.linalg.norm(output[i].numpy() - embed(word).numpy())
        mean /= len(indices)
        if mean < dist:
            nearest_word = word
            dist = mean
    return "" if nearest_word is None else nearest_word


def show_first_n(data, class_sample, cluster, n=10, nearest=None, most_frequent=None):
    result = [{"phrase": data[i], "cluster": str(cluster.labels_[i]), "nearest_word": nearest,
               "nearest_words": most_frequent} for i in range(len(cluster.labels_)) if
              cluster.labels_[i] == class_sample][:n]
    return result


def process_data(data, uid, n_clusters, max_clusters):
    if max_clusters is None and n_clusters is None:
        jobs[uid] = send_message("Must exist a query with max_clusters or n_clusters", label="error")
        return
    try:
        data = [phrase.replace("\n", "") for phrase in data]
        data = [phrase.replace("\"", "") for phrase in data]
        data = [phrase for phrase in data if len(phrase) != 0]
        data = [phrase for phrase in data if not phrase.isnumeric()]
        output = to_embed_vector(data)
        optimizer = gap_statistic.OptimalK(parallel_backend='joblib')
        if max_clusters is not None:
            max_clusters = int(max_clusters)
            n_clusters = optimizer(output.numpy(), cluster_array=[i for i in range(2, max_clusters)])
        elif n_clusters is not None:
            n_clusters = int(n_clusters)
        cluster = KMeans(n_clusters=n_clusters, random_state=0).fit(output.numpy())

        result = []
        for i in set(cluster.labels_):
            most_freq = most_frequent_words(data, i, cluster)
            nearest = nearest_words(data, output, i, cluster)
            result += show_first_n(data, i, cluster, nearest=nearest, most_frequent=",".join(most_freq))
        sem.acquire()
        jobs[uid] = result
        sem.release()
    except Exception as e:
        jobs[uid] = send_message(str(e), label="error")


class Clustering(flask_restful.Resource):

    def get(self, job_id):
        if job_id in jobs.keys():
            result = jobs[job_id]
            del jobs[job_id]
            return send_message(result, label="data"), 200
        return send_message("Job not found"), 404

    def post(self, job_id=None):
        body = flask.request.json
        n_clusters = flask.request.args.get('n_clusters')
        max_clusters = flask.request.args.get('max_clusters')
        uid = str(uuid.uuid4())
        thread = Thread(target=process_data, args=(body["messages"], uid, n_clusters, max_clusters))
        thread.start()
        return send_message(uid, label="id"), 201
