#!/usr/bin/env python3
"""清理 pyannote 模型缓存"""
import os
import shutil

def clear_pyannote_cache():
    # 使用 huggingface_hub 获取正确的缓存路径
    try:
        from huggingface_hub import constants as hf_constants
        cache_dir = hf_constants.HF_HUB_CACHE
    except:
        # 回退到默认路径
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")

    if not os.path.exists(cache_dir):
        print(f"缓存目录不存在: {cache_dir}")
        return

    print(f"扫描缓存目录: {cache_dir}")

    deleted = []
    for folder in os.listdir(cache_dir):
        if "pyannote" in folder.lower():
            folder_path = os.path.join(cache_dir, folder)
            print(f"删除: {folder_path}")
            try:
                shutil.rmtree(folder_path)
                deleted.append(folder)
            except Exception as e:
                print(f"  删除失败: {e}")

    if deleted:
        print(f"\n✓ 已清理 {len(deleted)} 个 pyannote 缓存文件夹")
    else:
        print("\n没有找到 pyannote 缓存")

if __name__ == "__main__":
    clear_pyannote_cache()
