import modal
import os
import shutil

app = modal.App("cleanup")

# Reuse your volume
volume = modal.Volume.from_name("datasets-volume")

@app.function(
    volumes={"/root/datasets": volume},
)
def delete_pruned_model():
    target_path = "/root/datasets/Artifacts/tokenized_cache"

    if os.path.exists(target_path):
        print(f"🗑️ Deleting: {target_path}")
        shutil.rmtree(target_path)
        print("✅ Folder deleted successfully!")
    else:
        print("⚠️ Folder not found, nothing to delete.")

    # Commit changes to persistent storage
    volume.commit()


@app.local_entrypoint()
def main():
    delete_pruned_model.remote()