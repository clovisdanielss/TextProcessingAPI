from resources.clustering import Clustering
from resources.knowledge import Knowledge
from initialize import api, app
   

api.add_resource(Clustering, '/clustering/', '/clustering/<job_id>')
api.add_resource(Knowledge, '/knowledge/', '/knowledge/<job_id>')

if __name__ == "__main__":
    app.run(host="0.0.0.0")