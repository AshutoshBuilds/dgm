import sys
import os
from huggingface_hub import snapshot_download


def main():
    if len(sys.argv) != 3:
        print("Usage: python tools/download_hf_model.py <repo_id> <local_dir>")
        sys.exit(1)
    repo_id = sys.argv[1]
    local_dir = sys.argv[2]
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
    print("Downloaded to", os.path.abspath(local_dir))


if __name__ == "__main__":
    main()
