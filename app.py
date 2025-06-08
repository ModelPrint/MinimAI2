from fastapi import FastAPI
import gradio as gr
from llama_cpp import Llama
from huggingface_hub import snapshot_download

# Modell automatisch aus HF Hub ziehen (nur beim ersten Start)
MODEL_DIR = snapshot_download(repo_id="meta-llama/Llama-2-7b", revision="4bit", library_name="llama_cpp")
llama = Llama(model_path=f"{MODEL_DIR}/ggml-model-q4_0.bin", n_ctx=1024)

def infer(prompt: str) -> str:
    out = llama(prompt, max_tokens=128)
    return out["choices"][0]["text"]

app = FastAPI()
iface = gr.Interface(fn=infer, inputs="text", outputs="text", title="Meine Llama KI")
app = gr.mount_gradio_app(app, iface, path="/")
