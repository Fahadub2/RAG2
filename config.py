import os
from dotenv import load_dotenv
load_dotenv()

# Model Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "gpt2")
VOCAB_SIZE = int(os.getenv("VOCAB_SIZE", "50257"))
HIDDEN_SIZE = int(os.getenv("HIDDEN_SIZE", "768"))
NUM_LAYERS = int(os.getenv("NUM_LAYERS", "12"))
NUM_HEADS = int(os.getenv("NUM_HEADS", "12"))
INTERMEDIATE_SIZE = int(os.getenv("INTERMEDIATE_SIZE", "3072"))
MAX_POSITION_EMBEDDINGS = int(os.getenv("MAX_POSITION_EMBEDDINGS", "1024"))
DROPOUT = float(os.getenv("DROPOUT", "0.1"))

# Training Configuration
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "5e-5"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.01"))
WARMUP_STEPS = int(os.getenv("WARMUP_STEPS", "1000"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "100000"))
SAVE_STEPS = int(os.getenv("SAVE_STEPS", "5000"))
LOGGING_STEPS = int(os.getenv("LOGGING_STEPS", "100"))
GRADIENT_ACCUMULATION_STEPS = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "4"))
MAX_GRAD_NORM = float(os.getenv("MAX_GRAD_NORM", "1.0"))
FP16 = os.getenv("FP16", "true").lower() == "true"

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# Create directories
for d in [DATA_DIR, MODEL_DIR, CHECKPOINT_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# Model size presets - Now includes 1.25B for Arabic context
MODEL_SIZES = {
    "small": {"hidden_size": 768, "num_layers": 12, "num_heads": 12, "intermediate_size": 3072},
    "medium": {"hidden_size": 1024, "num_layers": 24, "num_heads": 16, "intermediate_size": 4096},
    "large": {"hidden_size": 1280, "num_layers": 36, "num_heads": 20, "intermediate_size": 5120},
    "xl": {"hidden_size": 1600, "num_layers": 48, "num_heads": 25, "intermediate_size": 6400},
    "1b": {"hidden_size": 2048, "num_layers": 24, "num_heads": 16, "intermediate_size": 8192},
    "1.25b": {"hidden_size": 2304, "num_layers": 26, "num_heads": 18, "intermediate_size": 9216},
}

def get_model_config(size="1.25b"):
    """Get model configuration for specified size"""
    if size in MODEL_SIZES:
        return MODEL_SIZES[size]
    return MODEL_SIZES["1.25b"]

def calculate_parameters(hidden_size, num_layers, vocab_size, intermediate_size):
    """Calculate approximate number of parameters"""
    embedding = vocab_size * hidden_size
    attention = num_layers * (4 * hidden_size * hidden_size)
    ffn = num_layers * (3 * hidden_size * intermediate_size)
    output = hidden_size * vocab_size
    total = embedding + attention + ffn + output
    return total