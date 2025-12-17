#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基本功能测试脚本
用于验证主要模块可以正常导入和运行
"""

import sys
import os

def test_imports():
    """测试所有必需的模块是否可以导入"""
    print("=" * 60)
    print("测试模块导入")
    print("=" * 60)
    
    modules_to_test = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("soundfile", "SoundFile"),
        ("transformers", "Transformers"),
    ]
    
    success_count = 0
    for module, name in modules_to_test:
        try:
            __import__(module)
            print(f"✓ {name} 导入成功")
            success_count += 1
        except ImportError as e:
            print(f"✗ {name} 导入失败: {e}")
    
    print(f"\n导入成功: {success_count}/{len(modules_to_test)}")
    return success_count == len(modules_to_test)


def test_pipeline_imports():
    """测试自定义流程脚本是否可以导入"""
    print("\n" + "=" * 60)
    print("测试流程脚本")
    print("=" * 60)
    
    scripts = [
        "complete_pipeline",
        "unified_pipeline",
    ]
    
    success_count = 0
    for script in scripts:
        try:
            # 测试是否可以导入模块中的函数
            module = __import__(script)
            print(f"✓ {script}.py 可以导入")
            success_count += 1
        except Exception as e:
            print(f"✗ {script}.py 导入失败: {e}")
    
    print(f"\n脚本导入成功: {success_count}/{len(scripts)}")
    return success_count == len(scripts)


def test_audio_processing_functions():
    """测试音频处理基础函数"""
    print("\n" + "=" * 60)
    print("测试基础函数")
    print("=" * 60)
    
    try:
        from complete_pipeline import format_timestamp
        
        # 测试时间格式化
        result = format_timestamp(65.123)
        expected = "00:01:05,123"
        if result == expected:
            print(f"✓ format_timestamp 工作正常: {result}")
            return True
        else:
            print(f"✗ format_timestamp 结果不符: 期望 {expected}, 得到 {result}")
            return False
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


def test_file_structure():
    """测试文件结构"""
    print("\n" + "=" * 60)
    print("测试文件结构")
    print("=" * 60)
    
    required_files = [
        "complete_pipeline.py",
        "unified_pipeline.py",
        "README.md",
        "USAGE_GUIDE.md",
        "SOLUTION_SUMMARY.md",
        "requirements.txt",
    ]
    
    success_count = 0
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} 存在")
            success_count += 1
        else:
            print(f"✗ {file} 不存在")
    
    print(f"\n文件检查: {success_count}/{len(required_files)}")
    return success_count == len(required_files)


def main():
    """运行所有测试"""
    print("\n开始基本功能测试...\n")
    
    results = []
    
    # 测试文件结构
    results.append(("文件结构", test_file_structure()))
    
    # 测试导入
    results.append(("模块导入", test_imports()))
    
    # 测试流程脚本
    results.append(("流程脚本", test_pipeline_imports()))
    
    # 测试基础函数
    results.append(("基础函数", test_audio_processing_functions()))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status} - {name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n✅ 所有测试通过！系统基本功能正常。")
        return 0
    else:
        print(f"\n⚠️ {total - passed} 个测试失败，请检查依赖安装。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
