#!/usr/bin/env python3
"""
深度诊断 pyannote 模型加载问题
"""
import os
import sys
import tarfile
import io
import zipfile
from pathlib import Path

HF_TOKEN = os.environ.get("HF_TOKEN", "")

def get_cache_dir():
    """获取 HuggingFace 缓存目录"""
    return Path.home() / ".cache" / "huggingface" / "hub"

def find_model_files():
    """查找所有 pyannote 相关的模型文件"""
    cache_dir = get_cache_dir()
    print(f"缓存目录: {cache_dir}")
    print(f"目录存在: {cache_dir.exists()}")
    print()

    if not cache_dir.exists():
        print("缓存目录不存在，尝试手动下载模型...")
        return download_model_manually()

    # 查找所有 pyannote 相关目录
    pyannote_dirs = list(cache_dir.glob("models--pyannote*"))
    print(f"找到 {len(pyannote_dirs)} 个 pyannote 模型目录")

    model_files = []
    for model_dir in pyannote_dirs:
        print(f"\n模型: {model_dir.name}")

        # 查找 blobs 目录
        blobs_dir = model_dir / "blobs"
        if blobs_dir.exists():
            for blob in blobs_dir.iterdir():
                if blob.is_file():
                    size = blob.stat().st_size
                    print(f"  文件: {blob.name} ({size / 1024 / 1024:.2f} MB)")
                    if size > 1000000:  # > 1MB
                        model_files.append(blob)

        # 查找 snapshots 目录
        snapshots_dir = model_dir / "snapshots"
        if snapshots_dir.exists():
            for snapshot in snapshots_dir.iterdir():
                if snapshot.is_dir():
                    for f in snapshot.iterdir():
                        if f.suffix in ['.bin', '.pt', '.pth', '.safetensors']:
                            size = f.stat().st_size
                            print(f"  快照文件: {f.name} ({size / 1024 / 1024:.2f} MB)")
                            model_files.append(f)

    return model_files

def analyze_file(filepath):
    """分析文件格式"""
    print(f"\n分析文件: {filepath}")
    print(f"大小: {filepath.stat().st_size / 1024 / 1024:.2f} MB")

    with open(filepath, 'rb') as f:
        header = f.read(100)

    print(f"文件头 (hex): {header[:20].hex()}")

    # 检查是否是 ZIP 格式 (PyTorch 新格式)
    if header[:2] == b'PK':
        print("格式: ZIP (PyTorch 新格式)")
        try:
            with zipfile.ZipFile(filepath, 'r') as zf:
                print(f"ZIP 成员: {zf.namelist()[:10]}")
        except Exception as e:
            print(f"ZIP 读取失败: {e}")
        return "zip"

    # 检查是否是 TAR 格式 (PyTorch 旧格式)
    if tarfile.is_tarfile(filepath):
        print("格式: TAR (PyTorch 旧格式)")
        try:
            with tarfile.open(filepath, 'r') as tf:
                members = tf.getnames()
                print(f"TAR 成员: {members}")
                if 'storages' in members:
                    print("✓ 包含 'storages'")
                else:
                    print("✗ 缺少 'storages' - 文件可能损坏!")
        except Exception as e:
            print(f"TAR 读取失败: {e}")
        return "tar"

    # 检查是否是 pickle 格式
    if header[:2] == b'\x80\x02' or header[:2] == b'\x80\x04' or header[:2] == b'\x80\x05':
        print("格式: Pickle")
        return "pickle"

    # 检查是否是 safetensors
    if b'__metadata__' in header or header[:8] == b'safetens':
        print("格式: SafeTensors")
        return "safetensors"

    print("格式: 未知")
    return "unknown"

def download_model_manually():
    """手动下载模型文件"""
    print("=" * 60)
    print("手动下载 pyannote 模型")
    print("=" * 60)

    from huggingface_hub import hf_hub_download

    models_to_download = [
        ("pyannote/segmentation-3.0", "pytorch_model.bin"),
        ("pyannote/wespeaker-voxceleb-resnet34-LM", "pytorch_model.bin"),
    ]

    downloaded_files = []

    for repo_id, filename in models_to_download:
        print(f"\n下载 {repo_id}/{filename}...")
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                token=HF_TOKEN,
            )
            print(f"✓ 下载成功: {path}")
            downloaded_files.append(Path(path))
        except Exception as e:
            print(f"✗ 下载失败: {e}")

    return downloaded_files

def try_load_with_torch(filepath):
    """尝试用 torch 加载文件"""
    import torch

    print(f"\n尝试加载: {filepath}")

    # 方法1: 默认加载
    print("  方法1: torch.load(f, weights_only=False)")
    try:
        data = torch.load(filepath, weights_only=False)
        print(f"  ✓ 成功! 类型: {type(data)}")
        if isinstance(data, dict):
            print(f"    键: {list(data.keys())[:5]}")
        return True
    except Exception as e:
        print(f"  ✗ 失败: {e}")

    # 方法2: 用 map_location
    print("  方法2: torch.load(f, map_location='cpu', weights_only=False)")
    try:
        data = torch.load(filepath, map_location='cpu', weights_only=False)
        print(f"  ✓ 成功! 类型: {type(data)}")
        return True
    except Exception as e:
        print(f"  ✗ 失败: {e}")

    # 方法3: 用文件对象
    print("  方法3: 用文件对象加载")
    try:
        with open(filepath, 'rb') as f:
            data = torch.load(f, map_location='cpu', weights_only=False)
        print(f"  ✓ 成功! 类型: {type(data)}")
        return True
    except Exception as e:
        print(f"  ✗ 失败: {e}")

    return False

def main():
    print("=" * 60)
    print("Pyannote 模型深度诊断")
    print("=" * 60)
    print()

    # 1. 查找模型文件
    model_files = find_model_files()

    if not model_files:
        print("\n没有找到模型文件!")
        return

    # 2. 分析每个文件
    print("\n" + "=" * 60)
    print("文件格式分析")
    print("=" * 60)

    for f in model_files:
        format_type = analyze_file(f)

    # 3. 尝试加载
    print("\n" + "=" * 60)
    print("加载测试")
    print("=" * 60)

    for f in model_files:
        try_load_with_torch(f)

if __name__ == "__main__":
    main()
