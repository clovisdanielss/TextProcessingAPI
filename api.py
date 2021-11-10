import json
import flask
import flask_restful
from resources.clustering import Clustering

app = flask.Flask("ReportAPI")
api = flask_restful.Api(app)
   

api.add_resource(Clustering, '/clustering/','/clustering/<job_id>')

if __name__ == "__main__":
    app.run()