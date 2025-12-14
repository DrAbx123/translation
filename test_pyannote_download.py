#!/usr/bin/env python3
"""
测试 pyannote 模型下载
"""
import os

# HuggingFace Token
HF_TOKEN = os.environ.get("HF_TOKEN", "")

print("=" * 60)
print("测试 pyannote 模型下载")
print("=" * 60)
print()

# 检查 token
print(f"1. 使用 Token: {HF_TOKEN[:10]}...{HF_TOKEN[-4:]}")
print()

# 尝试下载
print("2. 尝试下载 pyannote/speaker-diarization-3.1 ...")
print()
print("   如果卡住或报错 401/403，说明：")
print("   - 需要先访问 https://huggingface.co/pyannote/speaker-diarization-3.1")
print("   - 登录 HuggingFace 账号")
print("   - 点击同意用户协议 (Accept license)")
print("   - 同样操作：https://huggingface.co/pyannote/segmentation-3.0")
print("   - 同样操作：https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM")
print()

try:
    from huggingface_hub import hf_hub_download, HfApi

    # 先测试 token 是否有效
    print("   测试 Token 有效性...")
    api = HfApi()
    try:
        user_info = api.whoami(token=HF_TOKEN)
        print(f"   ✓ Token 有效，用户: {user_info.get('name', 'Unknown')}")
    except Exception as e:
        print(f"   ✗ Token 无效: {e}")
        print()
        print("   请检查 HF_TOKEN 是否正确")
        exit(1)

    print()
    print("   下载模型配置文件...")

    # 尝试下载配置文件（小文件，测试权限）
    config_path = hf_hub_download(
        repo_id="pyannote/speaker-diarization-3.1",
        filename="config.yaml",
        token=HF_TOKEN,
    )
    print(f"   ✓ 配置文件下载成功: {config_path}")

    print()
    print("   下载分割模型...")
    seg_path = hf_hub_download(
        repo_id="pyannote/segmentation-3.0",
        filename="pytorch_model.bin",
        token=HF_TOKEN,
    )
    print(f"   ✓ 分割模型下载成功: {seg_path}")

    print()
    print("   下载嵌入模型...")
    emb_path = hf_hub_download(
        repo_id="pyannote/wespeaker-voxceleb-resnet34-LM",
        filename="pytorch_model.bin",
        token=HF_TOKEN,
    )
    print(f"   ✓ 嵌入模型下载成功: {emb_path}")

    print()
    print("=" * 60)
    print("✓ 所有模型下载成功！")
    print("=" * 60)
    print()
    print("现在可以重新运行 main_qwen.py")

except Exception as e:
    print()
    print(f"   ✗ 下载失败: {e}")
    print()
    print("   可能的原因：")
    print("   1. 没有同意 pyannote 模型的用户协议")
    print("   2. 网络问题（被墙）")
    print()
    print("   解决方案：")
    print("   1. 访问以下链接并同意协议：")
    print("      - https://huggingface.co/pyannote/speaker-diarization-3.1")
    print("      - https://huggingface.co/pyannote/segmentation-3.0")
    print("      - https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM")
    print()
    print("   2. 如果是网络问题，尝试设置镜像：")
    print("      $env:HF_ENDPOINT = 'https://hf-mirror.com'")
    print("      然后重新运行此脚本")
