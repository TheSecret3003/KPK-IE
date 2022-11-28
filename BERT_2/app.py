from flask import Flask, request
from get_synonim import get_synonim

app = Flask(__name__)

@app.route('/get-similar-entity-cosine/<version>', methods=['GET'])
def get_similar_entity(version):
    args = request.args
    nama_instansi = args.get('nama_instansi')
    similar_entity = get_synonim(nama_instansi, reference_version=version)
    
    return similar_entity

if __name__ == '__main__':
    app.run(debug=True)