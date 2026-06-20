import modal
import os

app = modal.App("cleanup-file")

# Reuse your volume
volume = modal.Volume.from_name("datasets-volume")

@app.function(
    volumes={"/root/datasets": volume},
)
def delete_json_file():
    target_files = [
        "log_2026-06-18_19-16-35.log",
    ]

    for target_file in target_files:
        target_file_path = os.path.join("/root/datasets/logs", target_file)

        if os.path.exists(target_file_path):
            print(f"🗑️ Deleting file: {target_file_path}")
            os.remove(target_file_path)
            print("✅ JSON file deleted successfully!")
        else:
            print(f"⚠️ File not found: {target_file_path}")

    # Commit once after all deletions (better)
    volume.commit()


@app.local_entrypoint()
def main():
    delete_json_file.remote()