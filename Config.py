# Contains all configuration options for the project


got_subs_dir = './Subs_Data_GOT'  # Unprocessed Data Dir
proc_data_dir = './data/' # Processed Data Dir
final_data = proc_data_dir + 'final_data/'
THRESHOLD = 2

# Real Vocab Sizes
# ENC_VOCAB = 8107
# DEC_VOCAB = 8090

#Cornell Vocab Sizes
ENC_VOCAB = 24426
DEC_VOCAB = 24674

OUTPUT_FILE = 'output_convo.txt'

BUCKETS = [(16, 19)]
PAD_ID = 0

NUM_SAMPLES = 512

PROCESSED_PATH = proc_data_dir

DATA_PATH = got_subs_dir

PROCESSED_PATH = final_data
CPT_PATH = 'checkpoints_1'

THRESHOLD = 2

PAD_ID = 0
UNK_ID = 1
START_ID = 2
EOS_ID = 3

TESTSET_SIZE = 25000


BUCKETS = [(16, 19)]

NUM_LAYERS = 3
HIDDEN_SIZE = 256
BATCH_SIZE = 64

LR = 0.5
MAX_GRAD_NORM = 5.0
