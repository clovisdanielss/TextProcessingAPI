import flask
import flask_restful
from threading import Semaphore
import tensorflow_hub as hub
from logging import INFO

jobs = {}
sem = Semaphore()
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
app = flask.Flask("ReportAPI")
app.logger.setLevel(INFO)
api = flask_restful.Api(app)