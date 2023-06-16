from torch.utils.data import Dataset
from transformers import GPT2Config, GPT2LMHeadModel, TrainingArguments, Trainer
import miditoolkit
import numpy as np


class PaddedDataset(Dataset):
    """
    Dataset class for managing and fetching tokenized and padded sequences.

    :param tokenizer: The tokenizer object responsible for tokenization
    :param data: List of input sequences
    :param max_length: The maximum length of the sequences after padding
    """
    def __init__(self, tokenizer, data, max_length):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        """
        Returns the total number of input sequences in the dataset.
        
        :return: The number of input sequences in the dataset
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Fetches a tokenized and padded input sequence along with its attention mask and labels.

        :param index: The index of the required sequence in the dataset
        :return: A dictionary containing input_ids, attention_mask and labels for the given sequence
        """
        inputs = self.tokenizer.encode_plus(
            self.data[index].split(" "),
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100  # Set padding tokens to -100 for language modeling
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    
    
def trainer_gpt2_transformer(hyperparameters, tokenizer, data, dirs):
    """
    Trains a GPT-2 transformer model with given hyperparameters, tokenizer and data.

    :param hyperparameters: Dictionary containing all the model and training hyperparameters
    :param tokenizer: Tokenizer used to encode the input sequences
    :param data: Dictionary containing the training and validation datasets
    :param dirs: Dictionary containing directories for saving model outputs and logs
    :return: Trainer object with a trained model
    """

    # define model config and model
    config = GPT2Config(
        vocab_size = tokenizer.vocab_size+3,
        n_positions = hyperparameters["max_length"], # max seq length
        n_embd = hyperparameters["emb_dim"],
        n_head = hyperparameters["attention_heads"], 
        n_layer = hyperparameters["layers"],
        dropout = hyperparameters["dropout"],
        bos_token_id = tokenizer.bos_token_id,
        eos_token_id = tokenizer.eos_token_id
    )

    model = GPT2LMHeadModel(config)

    # define training arguments
    training_args = TrainingArguments(
        output_dir = dirs["out"],
        overwrite_output_dir = True,
        num_train_epochs = hyperparameters["epochs"],
        per_device_train_batch_size = hyperparameters["batch_size"],
        save_steps = 1000,
        save_total_limit = 2, # maximum number of models to save
        learning_rate = hyperparameters["learning_rate"], 
        #weight_decay=0.01, # You can adjust the weight decay as needed
        #warmup_steps=1_000, # Number of warmup steps for learning rate scheduling
        seed = 4711,
        evaluation_strategy = "epoch",
        logging_strategy = "epoch"
    )

    # Create and train  Trainer
    trainer = Trainer(
        model = model,
        args = training_args,
        data_collator = data["data_collator"],
        train_dataset = data["train_dataset"],
        eval_dataset = data["eval_dataset"]
    )
    
    return trainer


def token_type(token, token_flags):
    """
    Determines the type of a given token based on the range it falls into.

    :param token: The token to be classified
    :param token_flags: Dictionary containing the start and end of each token type range
    :return: The type of the given token as a string
    """
    if token in range(token_flags["start_position_token"], token_flags["end_position_token"]+1):
        return "pos"
    elif token in range(token_flags["start_pitch_token"], token_flags["end_pitch_token"]+1):
        return "pitch"
    elif token in [token_flags["position_triole_1"], token_flags["position_triole_2"]]:
        return "ptriole"
    elif token in range(token_flags["start_duration_token"], token_flags["end_duration_token"]+1):
        return "duration"
    elif token == token_flags["duration_triole"]:
        return "dtriole"
    elif token == 0:
        return "Bar"
    else:
        print("Invalid token: {}".format(token))
        return "invalid"
        
NOTE_TYPES_following = {
    "start": ["pos"],
    "start-pos": ["pitch", "ptriole"],
    "start-pos-ptriole": ["pitch"],
    "start-pos-pitch": ["duration"],
    "start-pos-ptriole-pitch": ["duration"],
    "start-pos-pitch-duration": ["dtriole"],
    "start-pos-pitch-duration-dtriole": [],
    "start-pos-ptriole-pitch-duration": ["dtriole"],
    "start-pos-ptriole-pitch-duration-dtriole": [],
}

def analyze_token_sequence(seq, token_flags):
    """
    Analyzes a sequence of tokens and counts the number of each type of token sequence.

    :param seq: List of tokens
    :param token_flags: Dictionary containing the start and end of each token type range
    :return: Dictionary with counts of each type of token sequence
    """
    counts = {note_type: 0 for note_type in NOTE_TYPES_following}
    current_note_type = "start"

    for token in seq:

        if token_type(token, token_flags) in NOTE_TYPES_following[current_note_type]:
            current_note_type += "-" + token_type(token, token_flags)
        else:
            counts[current_note_type] += 1
            if token_type(token, token_flags) == "pos":
                current_note_type = "start-pos"
            else:
                current_note_type = "start"
    
    counts[current_note_type] += 1
    return counts


def predict(model, tokenizer, prompt="Bar_None", samples=5, max_length=100):
    """
    Generates a sequence of tokens based on a given prompt.

    :param model: Trained language model
    :param tokenizer: Tokenizer used to encode the input sequences
    :param prompt: Initial sequence to start generating from
    :param samples: Number of sequences to generate
    :param max_length: Maximum length of the generated sequences
    :return: List of generated sequences
    """
    # convert prompt into tensor
    inputs = tokenizer.encode(prompt.split(" "), return_tensors="pt")
    inputs = inputs.to("cpu")
    # generate output and convert to list
    outputs = model.generate(
        input_ids=inputs,
        max_length=max_length,  # maximum length of generated text
        do_sample=True,  # enable random sampling
        num_return_sequences=samples  # number of generated sequences
    )
    outputs = [[t.item() for t in list(tensor)] for tensor in outputs]
    #outputs = np.array(outputs)
    return outputs


######### back tokenizing #########

def token_to_event(tokens, token2word):
    """
    Converts a sequence of tokens into a sequence of musical events.

    :param tokens: List of tokens
    :param token2word: Dictionary mapping tokens to their corresponding words
    :return: List of dictionaries representing the musical events
    """
    events = []
    for token in tokens:
        event_name, event_value = token2word.get(token).split('_')
        events.append({
            "name": event_name,
            "time": None,
            "value": event_value,
            "text": None
        })
    return events

TICKS_PER_BEAT = 1024
TRIOLE_POS_1 = (TICKS_PER_BEAT/12).__round__()
TRIOLE_POS_2 = (TICKS_PER_BEAT/6).__round__()

def get_position_triole(flags, position, triole_position):
    """
    Determines the start time for a note based on its position bin and triplet position.

    :param flags: List of position bins in a bar
    :param position: Index of the note's position bin
    :param triole_position: Indicates whether the note is shifted to be a triplet (0 = no shift, 1 = one triplet shif, 2 = two triplet shift)
    :return: Start time for the note
    """
    if triole_position == 0:
        st = flags[position]
    elif triole_position == 1:
        st = flags[position] + TRIOLE_POS_1
    elif triole_position == 2:
        st = flags[position] + TRIOLE_POS_2
    return st


def write_midi(tokens, token2word, duration_bins, output_path):
    """
    Writes a sequence of tokens into a MIDI file.

    :param tokens: List of tokens
    :param token2word: Dictionary mapping tokens to their corresponding words
    :param duration_bins: List of possible duration bins for the notes
    :param output_path: Path to save the generated MIDI file
    :return: Number of incorrect notes found in the tokens sequence
    """
    events = token_to_event(tokens, token2word)
    # get downbeat and note (no time)
    incorrect_notes = 0
    temp_notes = []
    for i in range(len(events)-3):
        if events[i]["name"] == 'Bar' and i > 0:
            temp_notes.append('Bar')
        elif events[i]["name"] == 'Position':
            # get position bin 
            position = int(events[i]["value"].split('/')[0]) - 1
            # get triole position
            if events[i+1]["name"] == 'Position-Triole':
                triole_position = int(events[i+1]["value"])
                n = 1
            else:
                triole_position = 0
                n = 0
            if events[i+n+1]["name"] == 'Note-On' and \
            events[i+n+2]["name"] == 'Note-Duration' and events[i+n+2]["value"] != 'triole':
                # pitch
                pitch = int(events[i+n+1]["value"])
            else:
                incorrect_notes += 1
                continue
            # duration
            if i < len(events)-4 and events[i+n+3]["name"] == 'Note-Duration' and events[i+n+3]["value"] == 'triole':
                index = int(events[i+n+2]["value"])-1
                duration = int(duration_bins[index] / 3)
            else:
                index = int(events[i+n+2]["value"])-1
                duration = duration_bins[index]
            # adding
            temp_notes.append([position, triole_position ,pitch, duration])
    # get specific time for notes
    ticks_per_bar = TICKS_PER_BEAT * 4 # assume 4/4
    notes = []
    current_bar = 0
    for note in temp_notes:
        if note == 'Bar':
            current_bar += 1
        else:
            position, triole_position, pitch, duration = note
            # position (start time)
            current_bar_st = current_bar * ticks_per_bar
            current_bar_et = (current_bar + 1) * ticks_per_bar
            flags = np.linspace(current_bar_st, current_bar_et, 16, endpoint=False, dtype=int)
            st = get_position_triole(flags, position, triole_position)
            # duration (end time)
            et = st + duration
            notes.append(miditoolkit.Note(100, pitch, st, et))
    # write to midi
    midi = miditoolkit.midi.parser.MidiFile()
    midi.ticks_per_beat = TICKS_PER_BEAT
    # write instrument
    inst = miditoolkit.midi.containers.Instrument(0, is_drum=False)
    inst.notes = notes
    midi.instruments.append(inst)
        
    # write
    midi.dump(output_path)
    print("midi saved in {}".format(output_path))
    print("Number of incorrect notes: {}".format(incorrect_notes))
    return incorrect_notes


####### for predictions #######

def get_token_flags(vocab_config):
    """
    Generates a dictionary with the start and end of each token type range.

    :param vocab_config: Dictionary containing the vocabulary configuration
    :return: Dictionary containing the start and end of each token type range
    """
    pitch_range = vocab_config["pitch_range"]
    duration_steps = vocab_config["duration_steps"]
    token_flags = {
            "start_pitch_token": 1,
            "end_pitch_token": pitch_range,
            "start_duration_token": pitch_range+1,
            "end_duration_token": pitch_range+duration_steps,
        }

    if vocab_config["triole_tokens"]:
        token_flags.update({
            "duration_triole": pitch_range+duration_steps+1,
            "start_position_token": pitch_range+duration_steps+2,
            "end_position_token": pitch_range+duration_steps+17,
            "position_triole_1": pitch_range+duration_steps+18,
            "position_triole_2": pitch_range+duration_steps+19,
        })
    else:
        token_flags.update({
            "duration_triole": -100,
            "start_position_token": pitch_range+duration_steps+1,
            "end_position_token": pitch_range+duration_steps+16,
            "position_triole_1": -100,
            "position_triole_2": -100,
        })
    return token_flags