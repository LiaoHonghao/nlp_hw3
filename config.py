"""
Configuration file for machine translation project
"""

class Config:
    # Data paths
    TRAIN_LARGE_PATH = "./dataset/train_100k.jsonl"
    TRAIN_SMALL_PATH = "./dataset/train_10k.jsonl"
    VALID_PATH = "./dataset/valid.jsonl"
    TEST_PATH = "./dataset/test.jsonl"

    # Preprocessing
    MAX_LENGTH = 50  # Maximum sentence length
    MIN_FREQ = 2  # Minimum word frequency for vocabulary
    MAX_VOCAB_SIZE = 50000  # Maximum vocabulary size

    # Special tokens
    PAD_TOKEN = '<pad>'
    UNK_TOKEN = '<unk>'
    BOS_TOKEN = '<bos>'
    EOS_TOKEN = '<eos>'

    # RNN Model
    RNN_EMBED_DIM = 256
    RNN_HIDDEN_DIM = 512
    RNN_NUM_LAYERS = 2
    RNN_DROPOUT = 0.3
    RNN_CELL_TYPE = 'LSTM'  # 'LSTM' or 'GRU'
    ATTENTION_TYPE = 'dot'  # 'dot', 'multiplicative', 'additive'

    # Transformer Model
    TRANS_D_MODEL = 512
    TRANS_NHEAD = 8
    TRANS_NUM_ENCODER_LAYERS = 4
    TRANS_NUM_DECODER_LAYERS = 4
    TRANS_DIM_FEEDFORWARD = 2048
    TRANS_DROPOUT = 0.1
    POSITION_EMBEDDING = 'sinusoidal'  # 'sinusoidal', 'learned', 'relative'
    NORM_TYPE = 'LayerNorm'  # 'LayerNorm', 'RMSNorm'

    # Training
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 30
    GRAD_CLIP = 5.0
    TEACHER_FORCING_RATIO = 1.0  # 1.0 = always use teacher forcing
    LABEL_SMOOTHING = 0.1
    WARMUP_STEPS = 4000

    # Decoding
    BEAM_SIZE = 5
    MAX_DECODE_LENGTH = 60

    # Device
    DEVICE = 'cuda'  # 'cuda' or 'cpu'

    # Checkpoints
    RNN_CHECKPOINT_DIR = "./checkpoints/rnn"
    TRANS_CHECKPOINT_DIR = "./checkpoints/transformer"

    # Evaluation
    EVAL_EVERY = 1000  # Evaluate every N batches
    SAVE_EVERY = 5000  # Save checkpoint every N batches
