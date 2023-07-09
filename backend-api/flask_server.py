import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from models.base_model import Model
from models.gpt_neo import GptNeo

app = Flask(__name__)
CORS(app, origins=[os.getenv('FRONTEND_URL', 'http://localhost:3000')])

@app.route('/generate', methods=['POST'])
def generate():
    """
    Handle POST requests to the '/generate' endpoint. 
    Generates text using the model and prompt specified in the JSON body.

    Parameters
    ----------
    Expects JSON data in the request with the following attributes:
    model_name : str
        Name of the model to be used for inference. Currently only supports "GptNeo".
    prompt : str
        Prompt to be used as input.

    Returns
    -------
    flask.Response
        A JSON response with the generated output text.
    """
    data = request.get_json()  # get data from POST request

    # Check if prompt and model_name attributes exist in the request data
    if 'prompt' not in data or 'model_name' not in data:
        return jsonify({'error': 'Missing required data: prompt or model_name'}), 400

    prompt = data['prompt']
    model_name = data['model_name']
    
    if model_name == 'GptNeo':
        model = GptNeo()
    else:
        return jsonify({'error': 'Invalid model_name'}), 400

    output = model.generate(prompt)

    return jsonify({'output': output}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
