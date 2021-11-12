import flask
import flask_restful
from threading import Semaphore
import tensorflow_hub as hub

jobs = {}
sem = Semaphore()
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
app = flask.Flask("ReportAPI")
api = flask_restful.Api(app)