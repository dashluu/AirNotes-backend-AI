# AirNotes AI Backend

The backend of a basic note taking app inspired by Notion. There are still bugs regarding the UI and features so this web app should not be used in production. Sample screenshots are included below.

![img.png](screenshots/screenshot1.png)

![img.png](screenshots/screenshot2.png)

![img.png](screenshots/screenshot3.png)

![img.png](screenshots/screenshot4.png)

## Acknowledgement
* Stable diffusion with MLX: https://github.com/ml-explore/mlx-examples/tree/main/stable_diffusion/stable_diffusion
* Firebase reader: https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-firebase-realtimedb/llama_index/readers/firebase_realtimedb/base.py

## Requirement
* LlamaIndex and Ollama for LLM
* MLX for Stable Diffusion
* FastAPI for web server
* ChromaDB for vector database
* Firebase reader for syncing with remote Firebase server

## Usage
Clone the repository

```
git clone https://github.com/dashluu/AirNotes-backend-AI.git
```

Go to the project directory and run

```
uvicorn server:app --reload
```

The server can now accept request from the frontend. Check out my frontend app here https://github.com/dashluu/AirNotes-web

## Features
- [x] Generate summary using LLM and context from the current document
- [x] Generate answer to question using LLM and context from the current document
- [x] Semantic search across all documents using vector search and LLM
- [x] Generate images with Stable Diffusion(slow)

## Known issues
* Search is still limited, especially when filtering on the home page