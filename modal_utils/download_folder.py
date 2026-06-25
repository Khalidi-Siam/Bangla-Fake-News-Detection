import modal
import tarfile
import os

# Change only this
TARGET_FOLDER = "best_model"

app = modal.App("download")

# Reuse your volume
volume = modal.Volume.from_name("datasets-volume")


@app.function(
    volumes={"/root/datasets": volume},
    timeout=60 * 10,
)
def download_folder():
    source_dir = f"/root/datasets/Artifacts/{TARGET_FOLDER}"
    archive_path = f"/tmp/{TARGET_FOLDER}.tar.gz"

    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"{source_dir} not found!")

    print(f"📦 Compressing {TARGET_FOLDER}...")

    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(source_dir, arcname=TARGET_FOLDER)

    print("✅ Compression done")

    with open(archive_path, "rb") as f:
        return f.read()


@app.local_entrypoint()
def main():
    data = download_folder.remote()

    output_file = f"{TARGET_FOLDER}.tar.gz"

    with open(output_file, "wb") as f:
        f.write(data)

    print(f"✅ Download complete: {output_file}")