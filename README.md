# chatformers
Lean, model-agnostic framework for text generation models in the Huggingface transformers library. Includes a chat GUI and API server.

## Status
| Feature | Status |
| --- | --- |
| REST API | ✅ |
| Chat Frontend | ❌ |
| Transformers Model Support | ⚠️ |

⚠️ Currently only supports basic inference with GPT Neo.

## Running locally

### Backend
1. Install Python.
2. Terminal
```powershell
cd backend-api
pip install -r requirements.txt
flask --app flask_server run
```
