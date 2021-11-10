from util import send_message
import uuid
from threading import Thread, Semaphore
import flask
import flask_restful
from flask_restful import reqparse
import tensorflow_hub as hub
import numpy as np
import tensorflow_text
import nltk
from sklearn.cluster import KMeans
import tensorflow as tf

jobs = {}
sem = Semaphore()
parser = reqparse.RequestParser()
parser.add_argument('messages', required=True)
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")


class Clustering(flask_restful.Resource):
    def get(self, job_id):
        if job_id in jobs.keys():
            return send_message(jobs[job_id], label="data"), 200
        return send_message("Processo não encontrado."), 404

    def post(self, job_id=None):
        data = flask.request.json
        n_clusters = flask.request.args.get('n_clusters')
        if n_clusters is None:
            n_clusters = 20
        uid = str(uuid.uuid4())
        thread = Thread(target=self.process_data, args=(data["messages"], uid, int(n_clusters),))
        thread.start()
        return send_message(uid, label="id"), 201

    def process_data(self, data, uid, n_clusters):
        data = [phrase.replace("\n", "") for phrase in data]
        data = [phrase.replace("\"", "") for phrase in data]
        data = [phrase for phrase in data if len(phrase) != 0]
        data = [phrase for phrase in data if not phrase.isnumeric()]
        output = self.to_embed_vector(data)
        cluster = KMeans(n_clusters=n_clusters if len(data) > n_clusters else int(len(data) / 2), random_state=0)\
            .fit(output.numpy())
        result = []
        for i in set(cluster.labels_):
            most_freq = self.most_frequent_words(data, i, cluster)
            nearest = self.nearest_words(data, output, i, cluster)
            result += self.show_first_n(data, i, cluster, nearest=nearest, most_frequent=",".join(most_freq))
        sem.acquire()
        jobs[uid] = result
        sem.release()

    def to_embed_vector(self, data, batch=10, i=0, size=None):
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

    def most_frequent_words(self, data, class_sample, cluster, n=3):
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

    def nearest_words(self, data, output, class_sample, cluster, n=5):
        words = self.most_frequent_words(data, class_sample, cluster, n=n)
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

    def show_first_n(self, data, class_sample, cluster, n=10, nearest=None, most_frequent=None):
        result = [{"phrase": data[i], "cluster": str(cluster.labels_[i]), "nearest_word": nearest,
                   "nearest_words": most_frequent} for i in range(len(cluster.labels_)) if
                  cluster.labels_[i] == class_sample][:n]
        return result
