#!/bin/bash

# Django Setup Script
# This script helps set up the Django research platform

echo "========================================"
echo "Django Research Platform Setup"
echo "========================================"
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: No virtual environment detected!"
    echo "It's recommended to use a virtual environment."
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env

    # Generate encryption key
    echo "🔑 Generating encryption key..."
    ENCRYPTION_KEY=$(python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")

    # Update .env with generated key
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s|ENCRYPTION_KEY=your-generated-fernet-key-here|ENCRYPTION_KEY=$ENCRYPTION_KEY|" .env
    else
        # Linux
        sed -i "s|ENCRYPTION_KEY=your-generated-fernet-key-here|ENCRYPTION_KEY=$ENCRYPTION_KEY|" .env
    fi

    echo "✅ .env file created with encryption key"
    echo ""
else
    echo "✅ .env file already exists"
    echo ""
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt
echo ""

# Run migrations
echo "🗄️  Running database migrations..."
python manage.py makemigrations
python manage.py migrate
echo ""

# Create superuser prompt
echo "👤 Create superuser account?"
read -p "Create admin account? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python manage.py createsuperuser
fi

echo ""
echo "========================================"
echo "✅ Setup complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Run: python manage.py runserver"
echo "2. Open: http://localhost:8000"
echo "3. Sign up or login"
echo "4. Add your Anthropic API key in Settings"
echo "5. Start researching!"
echo ""
echo "Admin interface: http://localhost:8000/admin/"
echo ""
