import json
import flask
import flask_restful
from resources.clustering import Clustering
from resources.knowledge import Knowledge

app = flask.Flask("ReportAPI")
api = flask_restful.Api(app)
   

api.add_resource(Clustering, '/clustering/', '/clustering/<job_id>')
api.add_resource(Knowledge, '/knowledge/')

if __name__ == "__main__":
    app.run()