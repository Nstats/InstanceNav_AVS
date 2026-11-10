"""Optional Flask compatibility gateway. API clients can bypass this process."""
import argparse
import base64
from io import BytesIO
import os
import runpy
import sys

from flask import Flask, jsonify, request
from PIL import Image
from utils.model_client import get_model_client, ModelRequestError


def create_app():
    app = Flask(__name__)
    client = get_model_client()
    if client.backend != "openai":
        raise ValueError("API gateway requires PSGNAV_MODEL_BACKEND=openai")

    @app.get('/health')
    def health():
        return jsonify(backend='openai', model=client.model)

    @app.post('/chat')
    def chat():
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or not isinstance(data.get('prompt'), str) or not data['prompt'].strip():
            return jsonify(error='A nonempty prompt is required'), 400
        try:
            image = None
            if data.get('image'):
                with Image.open(BytesIO(base64.b64decode(data['image'], validate=True))) as source:
                    image = source.convert('RGB')
            return jsonify(response=client.complete(data['prompt'], image))
        except (ValueError, OSError, TypeError):
            return jsonify(error='Invalid image payload'), 400
        except ModelRequestError as exc:
            return jsonify(error=str(exc)), 502
    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5001)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--backend', choices=['openai', 'local'], default='openai')
    args = parser.parse_args()
    if args.backend == 'local':
        sys.argv = ['server_local.py', '--port', str(args.port), '--host', args.host]
        runpy.run_path(os.path.join(os.path.dirname(__file__), 'server_local.py'), run_name='__main__')
    else:
        os.environ['PSGNAV_MODEL_BACKEND'] = 'openai'
        create_app().run(host=args.host, port=args.port, threaded=True)
