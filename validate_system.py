#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistem Validation Script - YouTube Otomasyonu
✅ Tüm imports'ları kontrol eder
✅ Dependency'leri kontrol eder
✅ Configuration'ı validate eder
✅ Integration points'i test eder
"""
import sys
import importlib
from pathlib import Path

# Color codes for terminal
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_section(title):
    print(f"\n{BLUE}{'=' * 60}{RESET}")
    print(f"{BLUE}{title}{RESET}")
    print(f"{BLUE}{'=' * 60}{RESET}")

def print_success(msg):
    print(f"{GREEN}✅ {msg}{RESET}")

def print_error(msg):
    print(f"{RED}❌ {msg}{RESET}")

def print_warning(msg):
    print(f"{YELLOW}⚠️  {msg}{RESET}")

def print_info(msg):
    print(f"   {msg}")

def check_python_version():
    """Check Python version."""
    print_section("1. Python Version Check")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 7:
        print_success(f"Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_error(f"Python {version.major}.{version.minor} - Requires 3.7+")
        return False

def check_dependencies():
    """Check required dependencies."""
    print_section("2. Dependency Check")

    required = {
        'yaml': 'PyYAML',
        'PIL': 'Pillow',
        'requests': 'requests',
    }

    missing = []
    for module, package in required.items():
        try:
            importlib.import_module(module)
            print_success(f"{package}")
        except ImportError:
            print_error(f"{package} - NOT INSTALLED")
            missing.append(package)

    if missing:
        print_warning(f"Install missing packages: pip install {' '.join(missing)}")
        return False
    return True

def check_new_modules():
    """Check new enhancement modules."""
    print_section("3. New Enhancement Modules")

    modules = [
        'autoshorts.content.prompts.hook_patterns',
        'autoshorts.content.prompts.script_templates',
        'autoshorts.content.prompts.enhanced_prompts',
        'autoshorts.metadata.generator',
        'autoshorts.video.search_optimizer',
        'autoshorts.video.shot_variety',
        'autoshorts.audio.adaptive_mixer',
        'autoshorts.config.channel_config',
    ]

    all_ok = True
    for module in modules:
        try:
            importlib.import_module(module)
            print_success(module)
        except Exception as e:
            print_error(f"{module} - {str(e)}")
            all_ok = False

    return all_ok

def check_config_files():
    """Check configuration files."""
    print_section("4. Configuration Files")

    files = {
        'channels.yml': True,  # Required
        'autoshorts/config/config_manager.py': True,  # Required
    }

    all_ok = True
    for file, required in files.items():
        if Path(file).exists():
            print_success(file)
        else:
            if required:
                print_error(f"{file} - NOT FOUND (Required)")
                all_ok = False
            else:
                print_warning(f"{file} - NOT FOUND (Optional)")

    return all_ok

def test_integrations():
    """Test key integration points."""
    print_section("5. Integration Tests")

    tests_passed = 0
    tests_total = 0

    # Test 1: ConfigManager
    tests_total += 1
    try:
        from autoshorts.config.config_manager import ConfigManager
        config = ConfigManager(channel_name="test_channel")
        if hasattr(config.content, 'script_style'):
            print_success("ConfigManager with ScriptStyleConfig")
            tests_passed += 1
        else:
            print_warning("ConfigManager loaded but no script_style")
    except Exception as e:
        print_error(f"ConfigManager - {str(e)}")

    # Test 2: MetadataGenerator
    tests_total += 1
    try:
        from autoshorts.metadata.generator import MetadataGenerator
        gen = MetadataGenerator()
        print_success("MetadataGenerator instantiation")
        tests_passed += 1
    except Exception as e:
        print_error(f"MetadataGenerator - {str(e)}")

    # Test 3: SearchOptimizer
    tests_total += 1
    try:
        from autoshorts.video.search_optimizer import VideoSearchOptimizer
        optimizer = VideoSearchOptimizer()
        queries = optimizer.build_search_queries(
            sentence="The mountain rises above the clouds",
            keywords=["mountain", "clouds"],
            chapter_title="Geography",
            search_queries=None,
            sentence_type="content"
        )
        if len(queries) > 0:
            print_success(f"SearchOptimizer - Generated {len(queries)} queries")
            tests_passed += 1
        else:
            print_warning("SearchOptimizer - No queries generated")
    except Exception as e:
        print_error(f"SearchOptimizer - {str(e)}")

    # Test 4: ShotVariety
    tests_total += 1
    try:
        from autoshorts.video.shot_variety import ShotVarietyManager
        manager = ShotVarietyManager()
        plan = manager.plan_shot(
            sentence="The ocean waves crash against the shore",
            sentence_index=0,
            sentence_type="hook",
            total_sentences=10,
            keywords=["ocean", "waves"]
        )
        print_success(f"ShotVarietyManager - {plan.shot_type.value} shot planned")
        tests_passed += 1
    except Exception as e:
        print_error(f"ShotVarietyManager - {str(e)}")

    # Test 5: Caption Styling
    tests_total += 1
    try:
        from autoshorts.captions.karaoke_ass import get_random_style
        style = get_random_style(sentence_type="hook")
        print_success(f"Caption Styling - Style: {style}")
        tests_passed += 1
    except Exception as e:
        print_error(f"Caption Styling - {str(e)}")

    # Test 6: ChannelConfigLoader
    tests_total += 1
    try:
        from autoshorts.config.channel_config import get_channel_config_loader
        loader = get_channel_config_loader()
        channels = loader.list_channels()
        print_success(f"ChannelConfigLoader - {len(channels)} channels loaded")
        tests_passed += 1
    except Exception as e:
        print_error(f"ChannelConfigLoader - {str(e)}")

    print_info(f"\nTests Passed: {tests_passed}/{tests_total}")
    return tests_passed == tests_total

def check_backward_compatibility():
    """Check backward compatibility."""
    print_section("6. Backward Compatibility")

    # Test that get_random_style works with no args (backward compat)
    try:
        from autoshorts.captions.karaoke_ass import get_random_style
        style = get_random_style()  # No sentence_type argument
        print_success("get_random_style() backward compatible")
    except Exception as e:
        print_error(f"get_random_style() compatibility broken - {str(e)}")
        return False

    return True

def main():
    print(f"{BLUE}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║  YouTube Automation System - Validation Report            ║")
    print("║  Enhanced Features v2.0                                    ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(RESET)

    results = {
        'Python Version': check_python_version(),
        'Dependencies': check_dependencies(),
        'New Modules': check_new_modules(),
        'Config Files': check_config_files(),
        'Integrations': test_integrations(),
        'Backward Compatibility': check_backward_compatibility(),
    }

    # Summary
    print_section("Summary")
    passed = sum(results.values())
    total = len(results)

    for test, result in results.items():
        if result:
            print_success(f"{test}")
        else:
            print_error(f"{test}")

    print(f"\n{BLUE}{'─' * 60}{RESET}")
    if passed == total:
        print(f"{GREEN}🎉 ALL CHECKS PASSED ({passed}/{total}) - System Ready!{RESET}")
        return 0
    else:
        print(f"{YELLOW}⚠️  PARTIAL SUCCESS ({passed}/{total}) - Review errors above{RESET}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
