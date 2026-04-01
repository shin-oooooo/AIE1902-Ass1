
import os
from huggingface_hub import snapshot_download

def download_kronos_weights():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(base_dir, "kronos_weights")
    os.makedirs(weights_dir, exist_ok=True)

    models_to_download = [
        {"repo_id": "NeoQuasar/Kronos-small", "local_dir": "kronos-small"},
        {"repo_id": "NeoQuasar/Kronos-Tokenizer-base", "local_dir": "tokenizer-base"},
        {"repo_id": "NeoQuasar/Kronos-mini", "local_dir": "kronos-mini"},
        {"repo_id": "NeoQuasar/Kronos-Tokenizer-2k", "local_dir": "tokenizer-2k"},
        {"repo_id": "NeoQuasar/Kronos-base", "local_dir": "kronos-base"},
    ]

    for model in models_to_download:
        target_path = os.path.join(weights_dir, model["local_dir"])
        print(f"正在下载 {model['repo_id']} 到 {target_path}...")
        try:
            snapshot_download(
                repo_id=model["repo_id"],
                local_dir=target_path,
                local_dir_use_symlinks=False
            )
            print(f"✅ {model['repo_id']} 下载完成")
        except Exception as e:
            print(f"❌ {model['repo_id']} 下载失败: {e}")

if __name__ == "__main__":
    download_kronos_weights()
