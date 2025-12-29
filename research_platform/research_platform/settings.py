"""
Django settings for research_platform project.
"""

import os
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', 'django-insecure-change-this-in-production')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.environ.get('DJANGO_DEBUG', 'True') == 'True'

ALLOWED_HOSTS = os.environ.get('DJANGO_ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'agents.apps.AgentsConfig',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'research_platform.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'research_platform.wsgi.application'

# Database
# Use PostgreSQL in production, SQLite for local dev
DATABASES = {
    'default': {
        'ENGINE': os.environ.get('DB_ENGINE', 'django.db.backends.sqlite3'),
        'NAME': os.environ.get('DB_NAME', BASE_DIR / 'db.sqlite3'),
        'USER': os.environ.get('DB_USER', ''),
        'PASSWORD': os.environ.get('DB_PASSWORD', ''),
        'HOST': os.environ.get('DB_HOST', ''),
        'PORT': os.environ.get('DB_PORT', ''),
    }
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = 'static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_DIRS = [
    BASE_DIR / 'static',
]

# Media files (User-uploaded content)
MEDIA_URL = 'media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Authentication
LOGIN_URL = 'login'
LOGIN_REDIRECT_URL = 'dashboard'
LOGOUT_REDIRECT_URL = 'login'

# Encryption key for API keys
# CRITICAL: Set this in environment variables, never commit to version control
ENCRYPTION_KEY = os.environ.get('ENCRYPTION_KEY', None)
if not ENCRYPTION_KEY:
    # For development only - generate a key
    from cryptography.fernet import Fernet
    ENCRYPTION_KEY = Fernet.generate_key().decode()
    print(f"WARNING: Using auto-generated encryption key. Set ENCRYPTION_KEY in production!")
    print(f"Generated key: {ENCRYPTION_KEY}")

# Research Agent Settings
RESEARCH_AGENT_BASE_DIR = BASE_DIR.parent / 'research_agent'
RESEARCH_FILES_ROOT = MEDIA_ROOT / 'research_files'

# Memory Limits for Research Agent
# Automatically detects system RAM and defaults to 25% of total memory
# Can be overridden via RESEARCH_AGENT_MEMORY_LIMIT environment variable
# Set to 0 to disable memory limits

def _get_default_memory_limit():
    """Calculate default memory limit as 25% of total system RAM."""
    try:
        # Try using psutil (most reliable)
        import psutil
        total_memory = psutil.virtual_memory().total
        default_limit = int(total_memory * 0.25)
        print(f"Detected {total_memory / (1024**3):.1f}GB RAM, setting agent limit to {default_limit / (1024**3):.1f}GB (25%)")
        return default_limit
    except ImportError:
        # Fallback to platform-specific methods
        import platform
        import subprocess

        try:
            system = platform.system()
            if system == 'Darwin' or system == 'Linux':
                # macOS or Linux - use sysctl or /proc/meminfo
                if system == 'Darwin':
                    result = subprocess.run(['sysctl', 'hw.memsize'],
                                          capture_output=True, text=True, check=True)
                    total_memory = int(result.stdout.split(':')[1].strip())
                else:  # Linux
                    with open('/proc/meminfo', 'r') as f:
                        for line in f:
                            if line.startswith('MemTotal:'):
                                # MemTotal is in kB
                                total_memory = int(line.split()[1]) * 1024
                                break

                default_limit = int(total_memory * 0.25)
                print(f"Detected {total_memory / (1024**3):.1f}GB RAM, setting agent limit to {default_limit / (1024**3):.1f}GB (25%)")
                return default_limit
        except Exception as e:
            print(f"Warning: Could not detect system memory ({e}), defaulting to 8GB limit")

    # Final fallback
    return 8 * 1024 * 1024 * 1024  # 8GB

RESEARCH_AGENT_MEMORY_LIMIT = int(os.environ.get(
    'RESEARCH_AGENT_MEMORY_LIMIT',
    _get_default_memory_limit()
))

# Create necessary directories
os.makedirs(MEDIA_ROOT, exist_ok=True)
os.makedirs(RESEARCH_FILES_ROOT, exist_ok=True)
os.makedirs(STATIC_ROOT, exist_ok=True)
