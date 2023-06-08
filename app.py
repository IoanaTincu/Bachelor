from flask import Flask, request, jsonify
from flask_cors import CORS

from AlgorithmTextDocuments import OpDbscan

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes


@app.route('/', methods=['POST'])
def process_request():
    data = request.get_json()  # Get data sent in the request
    if data:
        epsilon = data.get('epsilon')
        n = data.get('n')
        minPts = data.get('minPts')
        numberFiles = data.get('numberFiles')

        # This block checks if any of the required parameters is not present in the request
        if not all([epsilon, n, minPts, numberFiles]):
            return jsonify({
                'message': 'One or more parameters missing in the request',
                'received_data': data,
            }), 400

        # You can now use epsilon, n, minPts, numberFiles here as per your logic
        algorithm = OpDbscan(epsilon, n, minPts, numberFiles)
        clusters = algorithm.OP_DBSCAN_Algorithm()
        print(algorithm.minimum_distance())

        indices = {}
        for i, point in enumerate(clusters):
            if point not in indices:
                indices[point] = [i]
            else:
                indices[point].append(i)

        response = {
            'indices': indices,
            'list': algorithm.samples
        }

        return jsonify(response), 200
    else:
        return jsonify({
            'message': 'No data provided in the request',
        }), 400


if __name__ == '__main__':
    app.run()
