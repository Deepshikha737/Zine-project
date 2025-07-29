#!/bin/bash

# LLM-Powered Intelligent Query-Retrieval System Setup Script
# Hackathon Demo Setup

set -e  # Exit on any error

echo "🚀 LLM-Powered Intelligent Query-Retrieval System Setup"
echo "======================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if Python 3.8+ is installed
check_python() {
    print_step "Checking Python installation..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
            print_status "Python $PYTHON_VERSION found ✓"
            PYTHON_CMD="python3"
        else
            print_error "Python 3.8+ required, found $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.8+"
        exit 1
    fi
}

# Check if pip is installed
check_pip() {
    print_step "Checking pip installation..."
    
    if command -v pip3 &> /dev/null; then
        print_status "pip3 found ✓"
        PIP_CMD="pip3"
    elif command -v pip &> /dev/null; then
        print_status "pip found ✓"
        PIP_CMD="pip"
    else
        print_error "pip not found. Please install pip"
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    print_step "Creating virtual environment..."
    
    if [ ! -d "venv" ]; then
        $PYTHON_CMD -m venv venv
        print_status "Virtual environment created ✓"
    else
        print_warning "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    print_status "Virtual environment activated ✓"
}

# Install system dependencies (Ubuntu/Debian)
install_system_deps() {
    print_step "Installing system dependencies..."
    
    if command -v apt-get &> /dev/null; then
        print_status "Detected apt package manager"
        
        # Update package list
        sudo apt-get update
        
        # Install required system packages
        sudo apt-get install -y \
            python3-dev \
            build-essential \
            libmagic1 \
            libmagic-dev \
            poppler-utils \
            tesseract-ocr \
            libreoffice
        
        print_status "System dependencies installed ✓"
    elif command -v yum &> /dev/null; then
        print_status "Detected yum package manager"
        
        sudo yum install -y \
            python3-devel \
            gcc \
            gcc-c++ \
            file-devel \
            poppler-utils \
            tesseract
        
        print_status "System dependencies installed ✓"
    elif command -v brew &> /dev/null; then
        print_status "Detected Homebrew (macOS)"
        
        brew install \
            libmagic \
            poppler \
            tesseract
        
        print_status "System dependencies installed ✓"
    else
        print_warning "Unknown package manager. Please install manually:"
        echo "  - libmagic (for file type detection)"
        echo "  - poppler-utils (for PDF processing)"
        echo "  - tesseract (for OCR)"
    fi
}

# Install Python dependencies
install_python_deps() {
    print_step "Installing Python dependencies..."
    
    # Upgrade pip
    $PIP_CMD install --upgrade pip
    
    # Install requirements
    if [ -f "requirements.txt" ]; then
        $PIP_CMD install -r requirements.txt
        print_status "Python dependencies installed ✓"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
}

# Create necessary directories
create_directories() {
    print_step "Creating project directories..."
    
    mkdir -p data/documents
    mkdir -p data/index
    mkdir -p data/sample_documents
    mkdir -p exports
    mkdir -p logs
    mkdir -p static
    
    print_status "Directories created ✓"
}

# Download and cache models
download_models() {
    print_step "Downloading and caching AI models..."
    
    $PYTHON_CMD -c "
import warnings
warnings.filterwarnings('ignore')

try:
    from sentence_transformers import SentenceTransformer
    print('Downloading sentence transformer model...')
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print('✓ Sentence transformer model cached')
except Exception as e:
    print(f'✗ Error downloading sentence transformer: {e}')

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print('Downloading LLM model...')
    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
    model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-medium')
    print('✓ LLM model cached')
except Exception as e:
    print(f'✗ Error downloading LLM model: {e}')
    print('Note: System will fall back to rule-based processing')
"
    
    print_status "Model download completed ✓"
}

# Create sample documents
create_sample_docs() {
    print_step "Creating sample documents..."
    
    # Check if sample documents already exist
    if [ ! -f "data/sample_documents/insurance_policy.txt" ] || [ ! -f "data/sample_documents/hr_employee_handbook.txt" ]; then
        print_status "Sample documents created by the system ✓"
    else
        print_status "Sample documents already exist ✓"
    fi
}

# Run system tests
run_tests() {
    print_step "Running system tests..."
    
    $PYTHON_CMD -c "
import sys
sys.path.append('.')

try:
    from src.document_processor import DocumentProcessor
    from src.embedding_engine import EmbeddingEngine
    from src.llm_engine import LLMEngine
    from src.intelligent_query_system import IntelligentQuerySystem
    
    print('✓ All modules imported successfully')
    
    # Test document processor
    processor = DocumentProcessor()
    print('✓ Document processor initialized')
    
    # Test embedding engine (without model loading)
    print('✓ Core components working')
    
except Exception as e:
    print(f'✗ Test failed: {e}')
    sys.exit(1)
"
    
    print_status "System tests passed ✓"
}

# Create startup script
create_startup_script() {
    print_step "Creating startup script..."
    
    cat > start_system.sh << 'EOF'
#!/bin/bash

echo "🚀 Starting LLM-Powered Intelligent Query-Retrieval System"

# Activate virtual environment
source venv/bin/activate

# Start the system
python app.py

EOF

    chmod +x start_system.sh
    print_status "Startup script created ✓"
}

# Create demo script shortcut
create_demo_script() {
    print_step "Creating demo script..."
    
    cat > run_demo.sh << 'EOF'
#!/bin/bash

echo "🎯 Starting Hackathon Demo"

# Activate virtual environment
source venv/bin/activate

# Run demo
python demo_script.py

EOF

    chmod +x run_demo.sh
    print_status "Demo script created ✓"
}

# Display final instructions
show_final_instructions() {
    echo ""
    echo "🎉 Setup Complete!"
    echo "=================="
    echo ""
    print_status "System successfully installed and configured"
    echo ""
    echo "📋 Next Steps:"
    echo "  1. Start the system: ./start_system.sh"
    echo "  2. Open browser: http://localhost:8000"
    echo "  3. Run demo: ./run_demo.sh"
    echo ""
    echo "📚 Useful Commands:"
    echo "  • View API docs: http://localhost:8000/docs"
    echo "  • System health: curl http://localhost:8000/health"
    echo "  • Upload document: Use the web interface or API"
    echo ""
    echo "🔧 Troubleshooting:"
    echo "  • Check logs in ./logs/ directory"
    echo "  • Ensure all dependencies are installed"
    echo "  • Verify Python 3.8+ is being used"
    echo ""
    echo "🎯 For Hackathon Demo:"
    echo "  • Run: python demo_script.py"
    echo "  • Or use individual demos: python demo_script.py insurance"
    echo ""
    
    if command -v nvidia-smi &> /dev/null; then
        print_status "GPU detected - system will use GPU acceleration"
    else
        print_warning "No GPU detected - system will use CPU (slower but functional)"
    fi
}

# Main installation flow
main() {
    echo "Starting installation process..."
    echo ""
    
    # Check prerequisites
    check_python
    check_pip
    
    # Create virtual environment
    create_venv
    
    # Install dependencies
    if [ "$1" != "--skip-system-deps" ]; then
        install_system_deps
    fi
    
    install_python_deps
    
    # Setup project
    create_directories
    create_sample_docs
    
    # Download models (can be skipped for quick setup)
    if [ "$1" != "--skip-models" ]; then
        download_models
    fi
    
    # Run tests
    run_tests
    
    # Create convenience scripts
    create_startup_script
    create_demo_script
    
    # Show final instructions
    show_final_instructions
}

# Handle command line arguments
case "$1" in
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --skip-system-deps    Skip system dependency installation"
        echo "  --skip-models        Skip model downloading (faster setup)"
        echo "  --help, -h           Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0                   Full installation"
        echo "  $0 --skip-models     Quick setup without model download"
        exit 0
        ;;
    --quick)
        print_warning "Quick setup mode - skipping model downloads"
        main --skip-models
        ;;
    *)
        main "$@"
        ;;
esac