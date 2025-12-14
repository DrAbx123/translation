#!/usr/bin/env python3
"""诊断 Python 环境问题"""
import sys
print("=" * 60)
print("Python 环境诊断")
print("=" * 60)

print(f"\nPython 路径: {sys.executable}")
print(f"Python 版本: {sys.version}")

print("\n模块搜索路径:")
for i, p in enumerate(sys.path[:5]):
    print(f"  {i}: {p}")

print("\n关键包位置:")
try:
    import torch
    print(f"  torch: {torch.__file__}")
    print(f"  torch 版本: {torch.__version__}")
except ImportError as e:
    print(f"  torch: 未安装 - {e}")

try:
    import pyannote.audio
    print(f"  pyannote.audio: {pyannote.audio.__file__}")
except ImportError as e:
    print(f"  pyannote.audio: 未安装 - {e}")

try:
    import lightning_fabric
    print(f"  lightning_fabric: {lightning_fabric.__file__}")
except ImportError as e:
    print(f"  lightning_fabric: 未安装 - {e}")

try:
    import whisperx
    print(f"  whisperx: {whisperx.__file__}")
except ImportError as e:
    print(f"  whisperx: 未安装 - {e}")

print("\n" + "=" * 60)
print("检查 torch 是否在虚拟环境中:")
print("=" * 60)

import os
venv_path = os.path.dirname(os.path.dirname(sys.executable))
expected_torch = os.path.join(venv_path, "Lib", "site-packages", "torch")

if os.path.exists(expected_torch):
    print(f"✓ torch 存在于虚拟环境: {expected_torch}")
else:
    print(f"✗ torch 不在虚拟环境中!")
    print(f"  期望路径: {expected_torch}")
    print()
    print("解决方案: 在虚拟环境中安装 torch")
    print("  .\.venv\Scripts\pip.exe install torch torchaudio --index-url https://download.pytorch.org/whl/cu121")
