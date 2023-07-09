import os, time
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from models.gpt_neo import GptNeo
from markupsafe import escape

# Cache for the model
model_cache = {
    'model': None,
    'model_name': None
}

def load_model(model_name):
    """
    Adds the specified model to the current cache and returns it.
    """
    start_time = time.time()
    # If a different model is requested, discard the current model and load the new one
    if model_cache['model_name'] != model_name:
        if model_name == 'GptNeo':
            model_cache['model'] = GptNeo()
            model_cache['model_name'] = model_name
        else:
            return jsonify({'error': 'Invalid model_name'}), 400

    model = model_cache['model']
    print(f"Model loading time: {time.time() - start_time} seconds")
    return model

print("Preloading GPT Neo...")
load_model("GptNeo")
print("Finished preloading GPT Neo.")

print("Starting up the flask server...")
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
    # Get data from POST request
    data = request.get_json()
    if 'prompt' not in data or 'model_name' not in data:
        return jsonify({'error': 'Missing required data: prompt or model_name'}), 400
    else:
        prompt = data['prompt']
        model_name = data['model_name']

    # Load the model
    model = load_model(model_name)

    # Generate the output
    start_time = time.time()
    output = model.generate(prompt)
    print(f"Generation time: {time.time() - start_time} seconds")

    return jsonify({'output': escape(output)}), 200

if __name__ == "__main__":
    print("Starting up flask server...")
    app.run(host="0.0.0.0", port=5000)
