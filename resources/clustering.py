from util import send_message
import uuid
from threading import Thread
import flask
import flask_restful
import numpy as np
import tensorflow_text
import nltk
from nltk.tokenize import RegexpTokenizer
from sklearn.cluster import KMeans, DBSCAN
import tensorflow as tf
import gap_statistic
from initialize import jobs, sem, embed, api, app
from sklearn.decomposition import PCA
from time import time
import logging
import gc


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


def most_frequent_words(data, cluster, n=3):
    frequency = {}
    for i in range(len(data)):
        phrase = data[i]
        class_sample = cluster.labels_[i]
        tokens = nltk.word_tokenize(phrase)
        for token in tokens:
            token = token.lower()
            class_in_dict = class_sample in frequency.keys()
            token_in_dict = class_in_dict and token in frequency[class_sample].keys()
            if token in nltk.corpus.stopwords.words("portuguese"):
                continue
            if not token.isalpha():
                continue
            if token_in_dict:
                frequency[class_sample][token] += 1
            elif class_in_dict:
                frequency[class_sample][token] = 1
            else:
                frequency[class_sample] = {token: 1}
    for class_sample in frequency.keys():
        frequency[class_sample] = sorted(frequency[class_sample].items(), key=lambda item: item[1], reverse=True)
    return dict(
        [(class_sample, [word for word, count in frequency[class_sample]][:n]) for class_sample in frequency.keys()])


def nearest_word(most_frequent, output, cluster, n=5):
    indices = {}
    result = {}
    for i in range(len(cluster.labels_)):
        if i in indices.keys():
            indices[i].append(cluster.labels_[i])
        else:
            indices[i] = [cluster.labels_[i]]
    for class_sample, words in most_frequent.items():
        dist = np.inf
        nearest_word = None
        for word in words:
            mean = 0
            if class_sample == -1:
                continue
            for i in indices[class_sample]:
                mean += np.linalg.norm(output[i] - embed(word).numpy())
            mean /= len(indices)
            if mean < dist:
                nearest_word = word
                dist = mean
        result[class_sample] = "" if nearest_word is None else nearest_word
    return result


def show_first_n(raw_data, processed_data, class_sample, cluster, n=1000, nearest=None, most_frequent=None):
    try:
        result = [{"phrase": raw_data[i], "cluster": int(cluster.labels_[i]), "nearest_word": nearest[class_sample],
                "frequent_words": most_frequent[class_sample]} for i in range(len(processed_data)) if
                cluster.labels_[i] == class_sample][:n]
    except:
        result = []
    return result

def run_kmeans(uid, n_clusters, max_clusters, output, algorithm):
    if max_clusters is not None:
        api.app.logger.info("Job %s - Searching optimal K", uid)
        optimizer = gap_statistic.OptimalK(parallel_backend='joblib')
        max_clusters = int(max_clusters)
        n_clusters = optimizer(output, cluster_array=[i for i in range(2, max_clusters)])
    elif n_clusters is not None:
        n_clusters = int(n_clusters)
    api.app.logger.info("Job %s - Running %s", uid, algorithm)
    cluster = KMeans(n_clusters=n_clusters, random_state=0).fit(output)
    return cluster

def run_dbscan(uid, distance_neighboor, output, algorithm):
    api.app.logger.info("Job %s - Running %s", uid, algorithm)
    cluster = DBSCAN(eps=distance_neighboor).fit(output)
    return cluster

def build_result(raw_data, processed_data, uid, cluster, output):
    api.app.logger.info("Job %s - Running Nearest Words", uid)
    result = []
    most_freq = most_frequent_words(processed_data, cluster)
    nearest = nearest_word(most_freq, output, cluster)
    for i in set(cluster.labels_):
        result += show_first_n(raw_data, processed_data, i, cluster, nearest=nearest, most_frequent=most_freq)
    return result

def process_data(data, uid, n_clusters, max_clusters, distance_neighboor, algorithm, n_components=64):
    time_start = time()
    if algorithm == "KMEANS" and max_clusters is None and n_clusters is None:
        jobs[uid] = send_message("Must exist a query with max_clusters or n_clusters", label="error")
        return
    try:
        tokenizer = RegexpTokenizer(r'\w+')
        urlTokenizer = RegexpTokenizer(r'https?:\/\/[\S]+')
        raw_data = []
        processed_data = []
        for raw_phrase in data:
            phrase = raw_phrase.replace("\n", "")
            phrase = phrase.replace("\"", "")
            if len(phrase) == 0 and phrase.isnumeric():
                continue
            phrase = phrase.lower()
            urls = urlTokenizer.tokenize(phrase)
            for url in urls:
                phrase = phrase.replace(url, "")
            tokens = [token for token in tokenizer.tokenize(phrase) if token not in nltk.corpus.stopwords.words("portuguese")]
            if len(tokens) <= 1:
                continue
            phrase = " ".join(tokens)
            processed_data.append(phrase)
            raw_data.append(raw_phrase)
        api.app.logger.info("Job %s - Transforming phrases in vectors", uid)
        output = to_embed_vector(processed_data).numpy()
        if algorithm == 'KMEANS':
            cluster = run_kmeans(uid, n_clusters, max_clusters, output, algorithm)
        elif algorithm == 'DBSCAN':
            cluster = run_dbscan(uid, distance_neighboor, output, algorithm)
        result = build_result(raw_data, processed_data, uid, cluster, output)
        sem.acquire()
        jobs[uid] = {"clusters": result, "cluster_centers": cluster.cluster_centers_.tolist() if algorithm == 'KMEANS' else []}
        sem.release()
        time_total = time() - time_start
        api.app.logger.info("Job %s - Done in %s", uid, str(time_total))
    except Exception as e:
        jobs[uid] = send_message(str(e), label="error")
        raise e


class Clustering(flask_restful.Resource):

    def get(self, job_id):
        if job_id in jobs.keys():
            result = jobs[job_id]
            del jobs[job_id]
            gc.collect()
            return send_message(result, label="data"), 200
        return send_message("Job not found"), 404

    def post(self, job_id=None):
        body = flask.request.json
        n_clusters = flask.request.args.get('n_clusters')
        max_clusters = flask.request.args.get('max_clusters')
        distance_neighboor = float(flask.request.args.get('distance_neighboor'))
        algorithm = flask.request.args.get('algorithm')
        if algorithm is None:
            algorithm = 'KMEANS'
        uid = str(uuid.uuid4())
        thread = Thread(target=process_data, args=(body["messages"], uid, n_clusters, max_clusters, distance_neighboor, algorithm))
        thread.start()
        return send_message(uid, label="id"), 201
