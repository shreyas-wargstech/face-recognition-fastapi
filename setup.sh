#!/bin/bash
# setup.sh - Quick setup script for LMS Face Recognition

echo "🚀 LMS Face Recognition - Quick Setup"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

# Check if Python 3.8+ is installed
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    print_status "Python $python_version detected"
else
    print_error "Python 3.8+ required. Found: $python_version"
    exit 1
fi

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    print_warning "Not in a virtual environment. Recommend creating one:"
    echo "  python3 -m venv face-recognition-venv"
    echo "  source face-recognition-venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    print_status "Virtual environment: $VIRTUAL_ENV"
fi

# Install system dependencies (Ubuntu/Debian)
if command -v apt-get &> /dev/null; then
    print_status "Installing system dependencies..."
    sudo apt-get update
    sudo apt-get install -y build-essential cmake pkg-config
    sudo apt-get install -y libjpeg-dev libtiff5-dev libpng-dev
    sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
    sudo apt-get install -y libxvidcore-dev libx264-dev
    sudo apt-get install -y libatlas-base-dev gfortran python3-dev
fi

# Install Python dependencies
print_status "Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    print_error "Failed to install Python packages"
    exit 1
fi

# Initialize database
print_status "Initializing database..."
python init_db.py --create-test-users --generate-key

if [ $? -ne 0 ]; then
    print_error "Database initialization failed"
    exit 1
fi

# Create uploads directory
mkdir -p uploads
print_status "Created uploads directory"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    print_status "Creating .env file..."
    cp .env.example .env
    
    # Generate encryption key
    encryption_key=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
    sed -i "s/your-face-encryption-key-32-bytes/$encryption_key/" .env
    
    print_warning "Please review and update the .env file with your settings"
fi

print_status "Setup completed successfully!"
echo ""
echo "🎯 Next steps:"
echo "1. Review your .env file settings"
echo "2. Start the API server:"
echo "   uvicorn main:app --reload"
echo "3. Open frontend.html in your browser"
echo "4. Test with the demo users:"
echo "   - Abhishek Jain (ID: 1)"
echo "   - Babita Sharma (ID: 2)"
echo "   - Chaitanya Kumar (ID: 3)"
echo "   - Deepak Kumar (ID: 4)"
echo ""
echo "📚 API Documentation: http://localhost:8000/docs"
echo "🏥 Health Check: http://localhost:8000/api/v1/health"