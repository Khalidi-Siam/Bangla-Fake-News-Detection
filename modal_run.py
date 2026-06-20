import modal
import sys
import os

app = modal.App("run")

volume = modal.Volume.from_name("datasets-volume", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11"
    )
    # Everything in ONE run_commands call — guarantees correct order,
    .run_commands(
        # System dependencies (needed for CUDA extensions)
        "apt-get update -y && apt-get install -y "
        "g++ build-essential git",

        # Core Python build tools
        "pip install --upgrade pip setuptools wheel packaging ninja",

        # Scientific stack
        "pip install "
        "pandas numpy scikit-learn matplotlib datasets tokenizers mlflow pydantic-settings",

        # PyTorch (CUDA 12.4 build)
        "pip install torch==2.4.0+cu124 --index-url https://download.pytorch.org/whl/cu124",

        # Transformers pinned for mamba-ssm compatibility
        "pip install transformers==4.40.0",

        # CUDA extension dependencies (order matters)
        "pip install causal-conv1d==1.4.0 --no-build-isolation",
        "pip install mamba-ssm==2.2.2 --no-build-isolation",
    )
    .add_local_dir("src",    remote_path="/app/src")
    .add_local_dir("config", remote_path="/app/config")
    .add_local_file("main.py", remote_path="/app/main.py")
)

@app.function(
    image=image,
    gpu="A100-40GB",
    cpu=6,
    memory=32768,
    volumes={"/root/datasets": volume},
    timeout=60 * 60 * 4,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run():
    import runpy
    sys.path.insert(0, "/app")
    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

    print("Starting pipeline via main.py...")
    runpy.run_path("/app/main.py", run_name="__main__")
    print("✅ Pipeline finished")

@app.local_entrypoint()
def main():
    run.remote()