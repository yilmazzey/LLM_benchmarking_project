#!/bin/bash

#################################################
# Automated Server Setup and Training Script
# For LLaMA Model Fine-tuning
#################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
REPO_URL="https://github.com/yilmazzey/LLM_benchmarking_project.git"
BRANCH="b200-finetune"
TARGET_DIR="$HOME/LLM_benchmarking_project"
TMUX_SESSION="llama-training"
PYTHON_VERSION="3.11"

# Default values
DEFAULT_MODELS="1b 3b 8b"
DEFAULT_EPOCHS=2
DEFAULT_BATCH_SIZE=8
DEFAULT_GRAD_ACCUM=8
DEFAULT_EVAL_SPLIT=0.1

MODELS="$DEFAULT_MODELS"
EPOCHS="$DEFAULT_EPOCHS"
BATCH_SIZE="$DEFAULT_BATCH_SIZE"
GRAD_ACCUM="$DEFAULT_GRAD_ACCUM"
EVAL_SPLIT="$DEFAULT_EVAL_SPLIT"

# Flags
FORCE_CLONE=false
SKIP_DEPS=false
CLEANUP=false
RESUME=false

#################################################
# Helper Functions
#################################################

print_header() {
    echo -e "${CYAN}================================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}================================================${NC}"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Automated server setup and training script for LLaMA fine-tuning.

IMPORTANT: This script uses PREPROCESSED datasets (processed_dataset/)
           created by data_process.py for fast parallel training.

Prerequisites:
    1. Preprocessed dataset must exist at: ./processed_dataset/
       Create it with: python data_process.py --input ./veri.json --output ./processed_dataset

    2. Or upload an existing processed_dataset/ directory to the repo root

Options:
    --clone                 Force fresh clone (delete and re-clone repository)
    --skip-deps             Skip dependency installation (if already done)
    --models "1b 3b 8b"     Models to train (default: "1b 3b 8b")
    --epochs N              Number of epochs (default: 2)
    --batch-size N          Per-device batch size (default: 8)
    --grad-accum N          Gradient accumulation steps (default: 8)
    --eval-split FLOAT      Evaluation split ratio (default: 0.1)
    --resume                Attach to existing tmux session
    --cleanup               Kill all training jobs and remove tmux session
    --help                  Show this help message

Default Training Settings:
    Epochs: 2
    Batch Size: 8
    Gradient Accumulation: 8 (effective batch size: 64)
    Eval Split: 0.1 (10% for validation)

Examples:
    # Full automated setup with default settings
    $0

    # Train only specific models
    $0 --models "1b 3b"

    # Custom training parameters
    $0 --epochs 5 --batch-size 4 --grad-accum 16

    # No evaluation split (train on all data)
    $0 --eval-split 0.0

    # Resume existing session
    $0 --resume

    # Cleanup everything
    $0 --cleanup

Training Commands:
    All training jobs automatically use: --use_processed --processed_path ./processed_dataset
    This ensures fast startup and no redundant preprocessing.

Tmux Commands:
    tmux attach -t $TMUX_SESSION    # Attach to session
    Ctrl+b then 0-5                 # Navigate windows
    Ctrl+b then d                   # Detach session

EOF
    exit 0
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        return 1
    fi
    return 0
}

#################################################
# Parse Arguments
#################################################

while [[ $# -gt 0 ]]; do
    case $1 in
        --clone)
            FORCE_CLONE=true
            shift
            ;;
        --skip-deps)
            SKIP_DEPS=true
            shift
            ;;
        --models)
            MODELS="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --grad-accum)
            GRAD_ACCUM="$2"
            shift 2
            ;;
        --eval-split)
            EVAL_SPLIT="$2"
            shift 2
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --cleanup)
            CLEANUP=true
            shift
            ;;
        --help|-h)
            show_usage
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            ;;
    esac
done

#################################################
# Cleanup Function
#################################################

cleanup_all() {
    print_header "Cleanup Mode"

    # Kill tmux session if exists
    if tmux has-session -t $TMUX_SESSION 2>/dev/null; then
        print_info "Killing tmux session: $TMUX_SESSION"
        tmux kill-session -t $TMUX_SESSION
        print_success "Tmux session killed"
    else
        print_warning "No tmux session found"
    fi

    # Kill any running training processes
    print_info "Checking for running training processes..."
    if pgrep -f "llama-ft.py" > /dev/null; then
        print_info "Killing training processes..."
        pkill -f "llama-ft.py"
        sleep 2
        print_success "Training processes killed"
    else
        print_info "No training processes found"
    fi

    print_success "Cleanup complete"
    exit 0
}

#################################################
# Resume Function
#################################################

resume_session() {
    print_header "Resume Mode"

    if tmux has-session -t $TMUX_SESSION 2>/dev/null; then
        print_info "Attaching to existing session: $TMUX_SESSION"
        tmux attach -t $TMUX_SESSION
    else
        print_error "No tmux session found: $TMUX_SESSION"
        print_info "Run without --resume to start a new session"
        exit 1
    fi
}

#################################################
# Main Execution
#################################################

# Handle cleanup mode
if [ "$CLEANUP" = true ]; then
    cleanup_all
fi

# Handle resume mode
if [ "$RESUME" = true ]; then
    resume_session
    exit 0
fi

print_header "LLaMA Fine-tuning Automation"
echo -e "${CYAN}Repository:${NC} $REPO_URL"
echo -e "${CYAN}Branch:${NC} $BRANCH"
echo -e "${CYAN}Target Directory:${NC} $TARGET_DIR"
echo ""
echo -e "${CYAN}Training Configuration:${NC}"
echo -e "  Models: ${GREEN}$MODELS${NC}"
echo -e "  Epochs: ${GREEN}$EPOCHS${NC}"
echo -e "  Batch Size: ${GREEN}$BATCH_SIZE${NC}"
echo -e "  Gradient Accumulation: ${GREEN}$GRAD_ACCUM${NC}"
echo -e "  Effective Batch Size: ${GREEN}$((BATCH_SIZE * GRAD_ACCUM))${NC}"
echo -e "  Eval Split: ${GREEN}$EVAL_SPLIT${NC} ($(echo "$EVAL_SPLIT * 100" | bc)% for validation)"
echo ""

#################################################
# Step 1: Clone Repository
#################################################

print_header "Step 1: Repository Setup"

if [ -d "$TARGET_DIR" ]; then
    if [ "$FORCE_CLONE" = true ]; then
        print_warning "Force clone enabled. Removing existing directory..."
        rm -rf "$TARGET_DIR"
        print_info "Cloning repository..."
        git clone -b "$BRANCH" "$REPO_URL" "$TARGET_DIR"
        print_success "Repository cloned"
    else
        print_info "Repository already exists at $TARGET_DIR"
        cd "$TARGET_DIR"
        print_info "Checking git status..."
        git fetch origin
        print_info "Pulling latest changes..."
        git pull origin "$BRANCH"
        print_success "Repository updated"
    fi
else
    print_info "Cloning repository..."
    git clone -b "$BRANCH" "$REPO_URL" "$TARGET_DIR"
    print_success "Repository cloned"
fi

cd "$TARGET_DIR"

#################################################
# Step 2: Install uv and Setup Python Environment
#################################################

print_header "Step 2: Python Environment Setup"

# Install uv if not present
if ! check_command uv; then
    print_info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add uv to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"

    print_success "uv installed"
else
    print_info "uv already installed"
fi

# Create venv if not exists
if [ ! -d ".venv" ]; then
    print_info "Creating Python $PYTHON_VERSION virtual environment..."
    uv venv --python $PYTHON_VERSION
    print_success "Virtual environment created"
else
    print_info "Virtual environment already exists"
fi

# Activate venv
source .venv/bin/activate
print_success "Virtual environment activated"

#################################################
# Step 3: Install Dependencies
#################################################

print_header "Step 3: Install Dependencies"

if [ "$SKIP_DEPS" = true ]; then
    print_warning "Skipping dependency installation (--skip-deps)"
else
    if [ -f "requirements.txt" ]; then
        print_info "Installing dependencies with uv..."
        uv pip install -r requirements.txt --index-strategy unsafe-best-match
        print_success "Dependencies installed"
    else
        print_error "requirements.txt not found!"
        exit 1
    fi
fi

#################################################
# Step 4: Validate Processed Dataset
#################################################

print_header "Step 4: Validate Processed Dataset"

if [ ! -d "processed_dataset" ]; then
    print_error "processed_dataset/ directory not found!"
    print_error ""
    print_error "This script expects a preprocessed dataset created by data_process.py"
    print_error ""
    print_info "To create the processed dataset:"
    echo -e "  ${GREEN}python data_process.py --input ./veri.json --output ./processed_dataset${NC}"
    print_error ""
    print_info "Or upload an existing processed_dataset directory to:"
    echo -e "  ${GREEN}$TARGET_DIR/processed_dataset${NC}"
    exit 1
fi

# Check for HuggingFace dataset format (dataset_info.json is the key file)
if [ ! -f "processed_dataset/dataset_info.json" ]; then
    print_error "Invalid processed_dataset format!"
    print_error "Missing dataset_info.json file (HuggingFace dataset format)"
    print_error ""
    print_info "Please create the dataset using:"
    echo -e "  ${GREEN}python data_process.py --input ./veri.json --output ./processed_dataset${NC}"
    exit 1
fi

# Check for data files
if [ ! -f "processed_dataset/data-00000-of-00001.arrow" ] && [ ! -d "processed_dataset/data" ]; then
    print_error "Dataset data files not found!"
    print_error "Expected: data-00000-of-00001.arrow or data/ directory"
    exit 1
fi

print_success "Processed dataset validated"

# Show dataset info if available
if [ -f "processed_dataset/dataset_info.json" ]; then
    SAMPLE_COUNT=$(grep -o '"num_examples": [0-9]*' processed_dataset/dataset_info.json | grep -o '[0-9]*')
    if [ ! -z "$SAMPLE_COUNT" ]; then
        print_info "Dataset samples: $SAMPLE_COUNT"
    fi
fi

#################################################
# Step 5: Setup tmux Training Sessions
#################################################

print_header "Step 5: Setup Training Sessions"

# Check if session already exists
if tmux has-session -t $TMUX_SESSION 2>/dev/null; then
    print_warning "Tmux session '$TMUX_SESSION' already exists"
    print_info "Use --resume to attach or --cleanup to remove"
    exit 1
fi

# Create tmux session
print_info "Creating tmux session: $TMUX_SESSION"
tmux new-session -d -s $TMUX_SESSION -n "status"

# Window 0: Status/Control Panel
tmux send-keys -t $TMUX_SESSION:0 "cd $TARGET_DIR" C-m
tmux send-keys -t $TMUX_SESSION:0 "source .venv/bin/activate" C-m
tmux send-keys -t $TMUX_SESSION:0 "clear" C-m
tmux send-keys -t $TMUX_SESSION:0 "echo '=== LLaMA Training Control Panel ==='" C-m
tmux send-keys -t $TMUX_SESSION:0 "echo 'Session: $TMUX_SESSION'" C-m
tmux send-keys -t $TMUX_SESSION:0 "echo 'Models: $MODELS'" C-m
tmux send-keys -t $TMUX_SESSION:0 "echo 'Epochs: $EPOCHS'" C-m
tmux send-keys -t $TMUX_SESSION:0 "echo ''" C-m
tmux send-keys -t $TMUX_SESSION:0 "echo 'Windows:'" C-m
tmux send-keys -t $TMUX_SESSION:0 "echo '  [0] Status (this window)'" C-m

# Create training windows for each model
WINDOW_NUM=1
for model in $MODELS; do
    print_info "Setting up training window for ${model} model..."

    tmux new-window -t $TMUX_SESSION:$WINDOW_NUM -n "train-${model}"
    tmux send-keys -t $TMUX_SESSION:$WINDOW_NUM "cd $TARGET_DIR" C-m
    tmux send-keys -t $TMUX_SESSION:$WINDOW_NUM "source .venv/bin/activate" C-m

    # Start training with nohup and tail -f in split pane
    LOG_FILE="train_${model}.log"
    tmux send-keys -t $TMUX_SESSION:$WINDOW_NUM "clear" C-m
    tmux send-keys -t $TMUX_SESSION:$WINDOW_NUM "echo '=== Training ${model} Model ==='" C-m
    tmux send-keys -t $TMUX_SESSION:$WINDOW_NUM "echo 'Starting training...'" C-m
    tmux send-keys -t $TMUX_SESSION:$WINDOW_NUM "echo 'Log file: $LOG_FILE'" C-m
    tmux send-keys -t $TMUX_SESSION:$WINDOW_NUM "echo ''" C-m

    # Start training in background and tail the log (with all parameters)
    tmux send-keys -t $TMUX_SESSION:$WINDOW_NUM "nohup python llama-ft.py --model ${model} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --gradient_accumulation ${GRAD_ACCUM} --eval_split ${EVAL_SPLIT} --use_processed --processed_path ./processed_dataset > $LOG_FILE 2>&1 &" C-m
    tmux send-keys -t $TMUX_SESSION:$WINDOW_NUM "sleep 2" C-m
    tmux send-keys -t $TMUX_SESSION:$WINDOW_NUM "tail -f $LOG_FILE" C-m

    # Update status window
    tmux send-keys -t $TMUX_SESSION:0 "echo '  [$WINDOW_NUM] train-${model} - Training ${model} model'" C-m

    WINDOW_NUM=$((WINDOW_NUM + 1))
done

# Window N: GPU Monitoring
print_info "Setting up GPU monitoring window..."
tmux new-window -t $TMUX_SESSION:$WINDOW_NUM -n "gpu"
tmux send-keys -t $TMUX_SESSION:$WINDOW_NUM "cd $TARGET_DIR" C-m
tmux send-keys -t $TMUX_SESSION:$WINDOW_NUM "watch -n 2 nvidia-smi" C-m
tmux send-keys -t $TMUX_SESSION:0 "echo '  [$WINDOW_NUM] gpu - GPU monitoring (nvidia-smi)'" C-m
WINDOW_NUM=$((WINDOW_NUM + 1))

# Window N+1: Combined Log Viewer
print_info "Setting up combined log viewer..."
tmux new-window -t $TMUX_SESSION:$WINDOW_NUM -n "logs"
tmux send-keys -t $TMUX_SESSION:$WINDOW_NUM "cd $TARGET_DIR" C-m
tmux send-keys -t $TMUX_SESSION:$WINDOW_NUM "clear" C-m
tmux send-keys -t $TMUX_SESSION:$WINDOW_NUM "echo '=== Combined Log Viewer ==='" C-m
tmux send-keys -t $TMUX_SESSION:$WINDOW_NUM "echo 'Waiting for log files...'" C-m
tmux send-keys -t $TMUX_SESSION:$WINDOW_NUM "sleep 3" C-m

# Build tail command for all logs
TAIL_CMD="tail -f"
for model in $MODELS; do
    TAIL_CMD="$TAIL_CMD train_${model}.log"
done
tmux send-keys -t $TMUX_SESSION:$WINDOW_NUM "$TAIL_CMD" C-m
tmux send-keys -t $TMUX_SESSION:0 "echo '  [$WINDOW_NUM] logs - Combined log viewer'" C-m

# Final status window updates
tmux send-keys -t $TMUX_SESSION:0 "echo ''" C-m
tmux send-keys -t $TMUX_SESSION:0 "echo 'Commands:'" C-m
tmux send-keys -t $TMUX_SESSION:0 "echo '  Ctrl+b then 0-$WINDOW_NUM   - Switch windows'" C-m
tmux send-keys -t $TMUX_SESSION:0 "echo '  Ctrl+b then d       - Detach session'" C-m
tmux send-keys -t $TMUX_SESSION:0 "echo '  Ctrl+c              - Stop tail (in log windows)'" C-m
tmux send-keys -t $TMUX_SESSION:0 "echo ''" C-m
tmux send-keys -t $TMUX_SESSION:0 "echo 'To reattach: tmux attach -t $TMUX_SESSION'" C-m
tmux send-keys -t $TMUX_SESSION:0 "echo 'To cleanup: $0 --cleanup'" C-m
tmux send-keys -t $TMUX_SESSION:0 "echo ''" C-m
tmux send-keys -t $TMUX_SESSION:0 "echo 'Training started! Check individual windows for progress.'" C-m

# Select status window
tmux select-window -t $TMUX_SESSION:0

print_success "Tmux sessions created and training started"

#################################################
# Completion
#################################################

print_header "Setup Complete!"
echo ""
print_success "All training jobs have been started in tmux session: $TMUX_SESSION"
print_success "Using preprocessed dataset: ./processed_dataset"
echo ""
print_info "Training Configuration:"
echo "  Models: $MODELS"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Gradient Accumulation: $GRAD_ACCUM"
echo "  Effective Batch Size: $((BATCH_SIZE * GRAD_ACCUM))"
echo "  Eval Split: $EVAL_SPLIT ($(echo "$EVAL_SPLIT * 100" | bc)%)"
echo "  Mode: --use_processed (fast parallel training)"
echo ""
print_info "To attach to the session:"
echo -e "  ${GREEN}tmux attach -t $TMUX_SESSION${NC}"
echo ""
print_info "Window layout:"
echo "  [0] status     - Control panel"
WINDOW_NUM=1
for model in $MODELS; do
    echo "  [$WINDOW_NUM] train-${model}  - Training ${model} model + live log"
    WINDOW_NUM=$((WINDOW_NUM + 1))
done
echo "  [$WINDOW_NUM] gpu        - GPU monitoring"
WINDOW_NUM=$((WINDOW_NUM + 1))
echo "  [$WINDOW_NUM] logs       - Combined log viewer"
echo ""
print_info "Navigation:"
echo "  Ctrl+b then 0-$WINDOW_NUM  - Switch windows"
echo "  Ctrl+b then d      - Detach (training continues in background)"
echo ""
print_info "Log files:"
for model in $MODELS; do
    echo "  $TARGET_DIR/train_${model}.log"
done
echo ""
print_info "To cleanup everything:"
echo -e "  ${GREEN}$0 --cleanup${NC}"
echo ""
print_success "Training started with preprocessed dataset!"
