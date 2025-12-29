#!/usr/bin/env python
"""
Diagnostic script to check Django setup.

Run this before starting the server to verify everything is configured correctly.
"""

import sys
import os
from pathlib import Path

print("=" * 60)
print("Django Research Platform - Setup Check")
print("=" * 60)
print()

# Check 1: Python version
print("✓ Checking Python version...")
python_version = sys.version_info
if python_version >= (3, 7):
    print(f"  ✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
else:
    print(f"  ❌ Python {python_version.major}.{python_version.minor} (need 3.7+)")
    sys.exit(1)
print()

# Check 2: Virtual environment
print("✓ Checking virtual environment...")
if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    print(f"  ✅ Virtual environment active: {sys.prefix}")
else:
    print("  ⚠️  No virtual environment detected (recommended but not required)")
print()

# Check 3: .env file
print("✓ Checking .env file...")
env_file = Path('.env')
if env_file.exists():
    print("  ✅ .env file exists")

    # Check for required variables
    with open(env_file) as f:
        env_content = f.read()

    required_vars = ['DJANGO_SECRET_KEY', 'ENCRYPTION_KEY']
    missing_vars = []

    for var in required_vars:
        if f'{var}=' in env_content:
            # Check if it's not the placeholder
            lines = [l for l in env_content.split('\n') if l.startswith(f'{var}=')]
            if lines:
                value = lines[0].split('=', 1)[1].strip()
                if 'your-' in value or 'change-' in value or 'here' in value:
                    missing_vars.append(f"{var} (needs to be updated)")
                else:
                    print(f"  ✅ {var} is set")
        else:
            missing_vars.append(var)

    if missing_vars:
        print(f"  ⚠️  These variables need to be set: {', '.join(missing_vars)}")
else:
    print("  ❌ .env file not found!")
    print("     Run: cp .env.example .env")
    print("     Then edit .env and set your keys")
print()

# Check 4: Django installation
print("✓ Checking Django installation...")
try:
    import django
    print(f"  ✅ Django {django.get_version()} installed")
except ImportError:
    print("  ❌ Django not installed!")
    print("     Run: pip install -r requirements.txt")
    sys.exit(1)
print()

# Check 5: Other dependencies
print("✓ Checking dependencies...")
required_packages = [
    ('cryptography', 'Encryption'),
    ('anthropic', 'Anthropic SDK'),
]

missing_packages = []
for package, description in required_packages:
    try:
        __import__(package)
        print(f"  ✅ {package} ({description})")
    except ImportError:
        print(f"  ❌ {package} ({description}) not installed")
        missing_packages.append(package)

if missing_packages:
    print()
    print("  Install missing packages:")
    print("    pip install -r requirements.txt")
    sys.exit(1)
print()

# Check 6: research_agent module
print("✓ Checking research_agent module...")
research_agent_path = Path('..') / 'research_agent'
if research_agent_path.exists():
    print(f"  ✅ research_agent directory found at: {research_agent_path.resolve()}")

    # Check for key files
    key_files = ['agent_api.py', 'data_structures.py', 'statistics.py']
    for file in key_files:
        file_path = research_agent_path / file
        if file_path.exists():
            print(f"  ✅ {file} exists")
        else:
            print(f"  ⚠️  {file} not found")
else:
    print("  ❌ research_agent directory not found!")
    print("     Expected at: {research_agent_path.resolve()}")
    print("     Make sure you're running this from the research_platform directory")
print()

# Check 7: Django settings
print("✓ Checking Django settings...")
try:
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'research_platform.settings')
    import django
    django.setup()
    from django.conf import settings

    print("  ✅ Django settings loaded")

    # Check ENCRYPTION_KEY
    if hasattr(settings, 'ENCRYPTION_KEY') and settings.ENCRYPTION_KEY:
        print("  ✅ ENCRYPTION_KEY configured")
    else:
        print("  ❌ ENCRYPTION_KEY not configured in settings")

    # Check RESEARCH_AGENT_BASE_DIR
    if hasattr(settings, 'RESEARCH_AGENT_BASE_DIR'):
        research_dir = settings.RESEARCH_AGENT_BASE_DIR
        if Path(research_dir).exists():
            print(f"  ✅ RESEARCH_AGENT_BASE_DIR exists: {research_dir}")
        else:
            print(f"  ⚠️  RESEARCH_AGENT_BASE_DIR path doesn't exist: {research_dir}")

except Exception as e:
    print(f"  ❌ Error loading Django settings: {e}")
    sys.exit(1)
print()

# Check 8: Database
print("✓ Checking database...")
try:
    from django.db import connection
    with connection.cursor() as cursor:
        cursor.execute("SELECT 1")
    print("  ✅ Database connection successful")

    # Check if migrations are needed
    from django.db.migrations.executor import MigrationExecutor
    executor = MigrationExecutor(connection)
    plan = executor.migration_plan(executor.loader.graph.leaf_nodes())

    if plan:
        print(f"  ⚠️  {len(plan)} migration(s) need to be applied")
        print("     Run: python manage.py migrate")
    else:
        print("  ✅ All migrations applied")

except Exception as e:
    print(f"  ⚠️  Database check: {e}")
    print("     Run: python manage.py migrate")
print()

# Summary
print("=" * 60)
print("Setup Check Complete!")
print("=" * 60)
print()
print("If all checks passed, you can run:")
print("  python manage.py runserver")
print()
print("Then visit: http://localhost:8000")
print()
