from flask import Flask, request, jsonify
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/api/hello')
def hello():
    return {'message': 'Hello, World!'}

@app.route('/api/cluster', methods=['POST'])
def cluster_data():
    dataset = request.json.get('dataset')
    if dataset:
        # Convierte el dataset en un DataFrame de Pandas
        df = pd.DataFrame(dataset)

        # Realiza el clustering usando K-Means
        kmeans = KMeans(n_clusters=3)  # Define el número de clusters deseado
        kmeans.fit(df)

        # Agrega las etiquetas de los clusters al DataFrame
        df['cluster'] = kmeans.labels_

        # Retorna los resultados
        clustered_data = df.to_dict(orient='records')
        return jsonify(clustered_data)
    else:
        return jsonify({'error': 'No se proporcionó un dataset'}), 400



if __name__ == '__main__':
    app.run(debug=True)