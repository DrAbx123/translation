#!/usr/bin/env python3
"""
诊断和修复 pyannote 模型缓存问题
"""
import os
import tarfile
import shutil
from pathlib import Path

def get_cache_dir():
    """获取 HuggingFace 缓存目录"""
    # 优先使用环境变量
    cache = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if cache:
        return Path(cache) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"

def check_model_file(filepath):
    """检查模型文件是否完整"""
    try:
        with open(filepath, 'rb') as f:
            # 检查是否是 tar 文件
            if tarfile.is_tarfile(filepath):
                with tarfile.open(filepath, 'r') as tar:
                    members = tar.getnames()
                    print(f"  ✓ 是有效的 tar 文件")
                    print(f"    成员: {members[:5]}{'...' if len(members) > 5 else ''}")
                    if 'storages' in members:
                        print(f"    ✓ 包含 'storages'")
                        return True
                    else:
                        print(f"    ✗ 缺少 'storages' - 文件损坏或下载不完整！")
                        return False
            else:
                # 可能是 safetensors 或其他格式
                size = os.path.getsize(filepath)
                print(f"  ? 非 tar 格式，大小: {size / 1024 / 1024:.2f} MB")
                return True
    except Exception as e:
        print(f"  ✗ 检查失败: {e}")
        return False

def diagnose_pyannote_cache():
    """诊断 pyannote 缓存"""
    cache_dir = get_cache_dir()
    print(f"缓存目录: {cache_dir}")
    print()

    if not cache_dir.exists():
        print("缓存目录不存在！")
        return

    pyannote_dirs = list(cache_dir.glob("models--pyannote*"))

    if not pyannote_dirs:
        print("没有找到 pyannote 模型缓存")
        return

    corrupted = []

    for model_dir in pyannote_dirs:
        print(f"模型: {model_dir.name}")

        # 查找 .bin 文件
        blob_dir = model_dir / "blobs"
        if blob_dir.exists():
            for blob_file in blob_dir.iterdir():
                if blob_file.is_file() and blob_file.stat().st_size > 1000000:  # > 1MB
                    print(f"  检查: {blob_file.name} ({blob_file.stat().st_size / 1024 / 1024:.2f} MB)")
                    if not check_model_file(blob_file):
                        corrupted.append(model_dir)
                        break
        print()

    if corrupted:
        print("=" * 50)
        print("发现损坏的模型缓存！")
        print("=" * 50)
        print()
        response = input("是否删除这些损坏的缓存并重新下载？(y/n): ")
        if response.lower() == 'y':
            for d in corrupted:
                print(f"删除: {d}")
                shutil.rmtree(d, ignore_errors=True)
            print()
            print("✓ 已清理损坏的缓存")
            print("请重新运行 main_qwen.py，模型将自动重新下载")
    else:
        print("所有模型文件看起来正常")
        print()
        print("如果仍然有问题，可能是 PyTorch 2.8 与 pyannote-audio 的兼容性问题")
        print("请尝试以下解决方案之一：")
        print()
        print("1. 升级 pyannote-audio 到 GitHub 最新版本：")
        print("   pip install git+https://github.com/pyannote/pyannote-audio.git")
        print()
        print("2. 或者删除整个缓存目录重新下载：")
        print(f"   Remove-Item -Recurse -Force \"{cache_dir}\\models--pyannote*\"")

if __name__ == "__main__":
    diagnose_pyannote_cache()
