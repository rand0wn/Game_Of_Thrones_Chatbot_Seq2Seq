import os
import re
import fileinput
import random
import Config
from nltk.tokenize import moses
import nltk
import numpy as np

#  nltk.download('all')

clean_data = []  # Clean data
clean_str_data = ''  # Clean Data in String

# Pre-Processing Data
def __preprocessing__(got_subs_dir):
    # Go to each File in data dir
    for subdir, dirs, files in os.walk(got_subs_dir):
        for file in files:
            file_string = open(os.path.join(subdir, file)).read() # Read File

            new_spec_str = re.sub('[^a-zA-Z0-9\n\.]', ' ', file_string)  # Remove Special Characters
            new_num_str = re.sub(r'\w*\d\w*', '', new_spec_str)  # Remove Numbers

            clean_data.append(new_num_str)

            # Write to String
            for i in range(0, len(clean_data)):
                temp_str_data = clean_str_data
                clean_str_data = temp_str_data + clean_data[i]

            # Write String to File
            with open('pre_spaces.txt', 'w') as f:
                f.write(clean_str_data)

            # Remove empty Lines from File in Between
            for line in fileinput.FileInput("pre_spaces.txt", inplace=1):
                if line.rstrip():
                    print line

# __preprocessing__(got_subs_dir)

# I have copied new file to 'pre_no_spaces.txt', others files are also present,
# also manually removed some noise, so make a backup before performing any operations on it.

# Divide dataset into train and test,
# both for encoding and decoding, For two way communication
# we need data to train on both ends(sender and receiver)
def __dataset_subs__(prep_filename):

    print "**********Dataset Preparation Start*************"

    # Read File into Lines
    with open(prep_filename, 'r') as f:
        lines = f.read().splitlines()

    # Read Lines without newline
    list_lines = filter(None, lines)

    # Generate random indexes for test data for 16% data set
    test_idx = random.sample(range(0, len(list_lines)), len(list_lines)/6)

    # Take 2 lines each for encoder and 2 each for decoder from list for conversation link buildup
    encoder_lines = open(Config.proc_data_dir + 'train_encoder', 'w')
    decoder_lines = open(Config.proc_data_dir + 'train_decoder', 'w')
    test_encoder_lines = open(Config.proc_data_dir + 'test_encoder', 'w')
    test_decoder_lines = open(Config.proc_data_dir + 'test_decoder', 'w')
    enc_dec_flag = 1  # To take alternate values in encoder, decoder

    for line_idx in xrange(0, len(list_lines)-1, 2):
        if line_idx in test_idx:  # If index in test data
            if enc_dec_flag == 0:
                #print "Dec test Combination", line_idx, line_idx + 1
                test_encoder_lines.write("%s\n" % list_lines[line_idx] + list_lines[line_idx + 1])
                #print "Dec: ", line_idx
                enc_dec_flag = 1
            else:
                #print "Enc test Combination", line_idx, line_idx + 1
                test_decoder_lines.write("%s\n" % list_lines[line_idx] + list_lines[line_idx + 1])
                #print "Enc: ", line_idx
                enc_dec_flag = 0
        else: # If index not in test data
            if enc_dec_flag == 0:
                #print "Dec train Combination", line_idx, line_idx + 1
                encoder_lines.write("%s\n" % list_lines[line_idx] + list_lines[line_idx + 1])
                #print "Dec: ", line_idx
                enc_dec_flag = 1
            else:
                #print "Enc train Combination", line_idx, line_idx + 1
                decoder_lines.write("%s\n" % list_lines[line_idx] + list_lines[line_idx + 1])
                #print "Enc: ", line_idx
                enc_dec_flag = 0

    print "**********Dataset Preparation Complete**********"

# Dataset from book data
def __dataset_books__(prep_filename):

    print "**********Dataset Preparation Start(BKS)*************"

    # Read File into Lines
    with open(prep_filename, 'r') as f:
        lines = f.read().splitlines()

    # Read Lines without newline
    list_lines = filter(None, lines)

    # Generate random indexes for test data for 16% data set
    test_idx = random.sample(range(0, len(list_lines)), len(list_lines)/6)

    # Take 2 lines each for encoder and 2 each for decoder from list for conversation link buildup
    encoder_lines = open(Config.proc_data_dir + 'train1_encoder', 'w')
    decoder_lines = open(Config.proc_data_dir + 'train1_decoder', 'w')
    test_encoder_lines = open(Config.proc_data_dir + 'test1_encoder', 'w')
    test_decoder_lines = open(Config.proc_data_dir + 'test1_decoder', 'w')
    enc_dec_flag = 1  # To take alternate values in encoder, decoder

    for line_idx in xrange(0, len(list_lines)-1, 1):
        if line_idx in test_idx:  # If index in test data
            if enc_dec_flag == 0:
                #print "Dec test Combination", line_idx, line_idx + 1
                test_encoder_lines.write("%s\n" % list_lines[line_idx])
                #print "Dec: ", line_idx
                enc_dec_flag = 1
            else:
                #print "Enc test Combination", line_idx, line_idx + 1
                test_decoder_lines.write("%s\n" % list_lines[line_idx])
                #print "Enc: ", line_idx
                enc_dec_flag = 0
        else: # If index not in test data
            if enc_dec_flag == 0:
                #print "Dec train Combination", line_idx, line_idx + 1
                encoder_lines.write("%s\n" % list_lines[line_idx])
                #print "Dec: ", line_idx
                enc_dec_flag = 1
            else:
                #print "Enc train Combination", line_idx, line_idx + 1
                decoder_lines.write("%s\n" % list_lines[line_idx])
                #print "Enc: ", line_idx
                enc_dec_flag = 0

    print "**********Dataset Preparation Complete(BKS)**********"


# Call dataset books function
#(__dataset_books__(Config.proc_data_dir + 'bk_proc_text.txt'))

# Call dataset function
#(__dataset_subs__(Config.proc_data_dir + 'pre_no_spaces.txt'))


# Now merge both files from books and subs, I have merged it manually in folder in final_data.


# Create Vocab for each decoder and encoder files
def __create_conv_vocab__(filename):
    in_path = os.path.join(Config.final_data, filename)
    out_path = os.path.join(Config.final_data, 'vocab.{}'.format(filename[-3:]))

    tokenizer = moses.MosesTokenizer()  # For Tokens

    vocab = {}

    with open(in_path, 'rb') as f:
        for line in f.readlines():
            for token in tokenizer.tokenize(line.decode('utf-8')):
                # Create Vocab Dictionary
                if not token in vocab:
                    vocab[token] = 0
                vocab[token] += 1

    sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)

    # Write Vocab To File
    with open(out_path, 'wb') as f:
        f.write('<pad>' + '\n')
        f.write('<unk>' + '\n')
        f.write('<s>' + '\n')
        f.write('<\s>' + '\n')
        index = 4
        for word in sorted_vocab:
            if vocab[word] < Config.THRESHOLD:
                with open('Config.py', 'ab') as cf:
                    if filename[-3:] == 'enc':
                        cf.write('ENC_VOCAB = ' + str(index) + '\n')
                    else:
                        cf.write('DEC_VOCAB = ' + str(index) + '\n')
                break
            f.write(word.encode('utf-8') + '\n')
            index += 1

# __create_conv_vocab__('TRAIN.enc')
# __create_conv_vocab__('TRAIN.dec')

def load_vocab(vocab_path):
    with open(vocab_path, 'rb') as f:
        words = f.read().splitlines()
    return words, {words[i]: i for i in range(len(words))}

# Sentence as a combination of vocab ids
def sentence2id(vocab, line):
    tokenizer = moses.MosesTokenizer()
    return [vocab.get(token, vocab['<unk>']) for token in tokenizer.tokenize(line.decode('utf-8'))]

def __map_vocab__(data, mode):
    vocab_path = 'vocab.' + mode
    in_path = data + '.' + mode
    out_path = data + '_ids.' + mode

    _, vocab = load_vocab(os.path.join(Config.final_data, vocab_path))
    in_file = open(os.path.join(Config.final_data, in_path), 'rb')
    out_file = open(os.path.join(Config.final_data, out_path), 'wb')

    lines = in_file.read().splitlines()
    for line in lines:
        if mode == 'dec':  # we only care about '<s>' and </s> in encoder
            ids = [vocab['<s>']]
        else:
            ids = []
        ids.extend(sentence2id(vocab, line))
        # ids.extend([vocab.get(token, vocab['<unk>']) for token in basic_tokenizer(line)])
        if mode == 'dec':
            ids.append(vocab['<\s>'])
        out_file.write(' '.join(str(id_) for id_ in ids) + '\n')


# __map_vocab__('TRAIN', 'enc')
# __map_vocab__('TRAIN', 'dec')
# __map_vocab__('TEST', 'enc')
# __map_vocab__('TEST', 'dec')

# Next is creating buckets, buckets are a group of data points
# to be passed at once in rnn for one epoch(set of defined iterations)

def gen_buckets(enc, dec): # encoder and decoder file generated from map_vocab

    encode_file = open(os.path.join(Config.final_data, enc), 'rb')
    decode_file = open(os.path.join(Config.final_data, dec), 'rb')

    encode, decode = encode_file.readline(), decode_file.readline()
    data_buckets = [[] for _ in Config.BUCKETS]

    i = 0
    while encode and decode:
        if (i + 1) % 10000 == 0:
            print("Bucketing conversation number", i)

        encode_ids = [int(id_) for id_ in encode.split()]
        decode_ids = [int(id_) for id_ in decode.split()]

        for bucket_id, (encode_max_size, decode_max_size) in enumerate(Config.BUCKETS):

            if len(encode_ids) <= encode_max_size and len(decode_ids) <= decode_max_size:
                data_buckets[bucket_id].append([encode_ids, decode_ids])
                break

        encode, decode = encode_file.readline(), decode_file.readline()
        i += 1

    return data_buckets

# Prepare batch from buckets to feed into model

def _pad_input(input_, size):
    return input_ + [Config.PAD_ID] * (size - len(input_))

def _reshape_batch(inputs, size, batch_size):
    """ Create batch-major inputs. Batch inputs are just re-indexed inputs
    """
    batch_inputs = []
    for length_id in range(size):
        batch_inputs.append(np.array([inputs[batch_id][length_id]
                                    for batch_id in range(batch_size)], dtype=np.int32))
    return batch_inputs


def get_batch(data_bucket, bucket_id, batch_size=1):
    """ Return one batch to feed into the model """
    # only pad to the max length of the bucket
    encoder_size, decoder_size = Config.BUCKETS[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    for _ in range(batch_size):
        encoder_input, decoder_input = random.choice(data_bucket)
        # pad both encoder and decoder, reverse the encoder
        encoder_inputs.append(list(reversed(_pad_input(encoder_input, encoder_size))))
        decoder_inputs.append(_pad_input(decoder_input, decoder_size))

    # now we create batch-major vectors from the data selected above.
    batch_encoder_inputs = _reshape_batch(encoder_inputs, encoder_size, batch_size)
    batch_decoder_inputs = _reshape_batch(decoder_inputs, decoder_size, batch_size)

    # create decoder_masks to be 0 for decoders that are padding.
    batch_masks = []
    for length_id in range(decoder_size):
        batch_mask = np.ones(batch_size, dtype=np.float32)
        for batch_id in range(batch_size):
            # we set mask to 0 if the corresponding target is a PAD symbol.
            # the corresponding decoder is decoder_input shifted by 1 forward.
            if length_id < decoder_size - 1:
                target = decoder_inputs[batch_id][length_id + 1]
            if length_id == decoder_size - 1 or target == Config.PAD_ID:
                batch_mask[batch_id] = 0.0
        batch_masks.append(batch_mask)
    return batch_encoder_inputs, batch_decoder_inputs, batch_masks

def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass
