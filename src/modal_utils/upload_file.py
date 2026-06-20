import modal
import os
import shutil

local_file_name = "bert_class_weights.npy"  # your file name

app = modal.App(f"upload-{local_file_name.replace('.', '-')}")

# Reuse your existing volume
volume = modal.Volume.from_name("datasets-volume", create_if_missing=True)

# Build image and include local file
image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .add_local_file(
        local_file_name,
        remote_path=f"/tmp/{local_file_name}",
    )
)

@app.function(
    image=image,
    volumes={"/root/datasets": volume},
    timeout=60 * 20,
)
def upload_file():
    src = f"/tmp/{local_file_name}"
    dst = f"/root/datasets/{local_file_name}"

    print(f"📄 Uploading {local_file_name}...")

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)

    print(f"✅ Uploaded successfully to {dst}")

    volume.commit()


@app.local_entrypoint()
def main():
    upload_file.remote()