from threading import Semaphore
import tensorflow_hub as hub

jobs = {}
sem = Semaphore()
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")