import modal
import os
import shutil

local_folder_name = "tokenized_cache_bert" # you can change this to something more specific if you want
app = modal.App(f"upload-{local_folder_name}")

# Reuse your existing volume
volume = modal.Volume.from_name("datasets-volume", create_if_missing=True)

# Build image and include local specified folder
image = (
    modal.Image.debian_slim()
    .add_local_dir(local_folder_name, remote_path=f"/tmp/{local_folder_name}")  # 👈 your local folder
)

@app.function(
    image=image,
    volumes={"/root/datasets": volume},
    timeout=60 * 20,  # increase if many files
)
def upload_folders():
    local_path = f"/tmp/{local_folder_name}"
    remote_path = f"/root/datasets/{local_folder_name}"

    os.makedirs(remote_path, exist_ok=True)

    print("📂 Uploading...")

    # Copy everything recursively
    for item in os.listdir(local_path):
        src = os.path.join(local_path, item)
        dst = os.path.join(remote_path, item)

        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    print("✅Uploaded successfully!")

    # Verify
    for root, dirs, files in os.walk(remote_path):
        print(root, "->", len(files), "files")

    volume.commit()


@app.local_entrypoint()
def main():
    upload_folders.remote()