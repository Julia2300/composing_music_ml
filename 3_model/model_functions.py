from torch.utils.data import Dataset
from transformers import GPT2Config, GPT2LMHeadModel, TrainingArguments, Trainer
import miditoolkit
import numpy as np


class PaddedDataset(Dataset):
    def __init__(self, tokenizer, data, max_length):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
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
        logging_dir = dirs["logging"],
        logging_steps = 100,
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


def token_type(token):
    if token in range(102, 118):
        return "pos"
    elif token in range(1, 37):
        return "pitch"
    elif token in range(118, 120):
        return "ptriole"
    elif token in range(37, 101):
        return "duration"
    elif token == 101:
        return "dtriole"
    elif token == 0:
        return "Bar"
    elif token in range(120, 123):
        return "Bar"
    else:
        raise ValueError("Invalid token: {}".format(token))
        
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

def analyze_token_sequence(seq):
    counts = {note_type: 0 for note_type in NOTE_TYPES_following}
    current_note_type = "start"

    for token in seq:

        if token_type(token) in NOTE_TYPES_following[current_note_type]:
            current_note_type += "-" + token_type(token)
        else:
            counts[current_note_type] += 1
            if token_type(token) == "pos":
                current_note_type = "start-pos"
            else:
                current_note_type = "start"
    
    counts[current_note_type] += 1
    return counts


def predict(model, tokenizer, prompt="Bar_None", samples=5, max_length=100):
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


def token_to_event(tokens, token2word):
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

def get_position_triole(flags, position, triole_position):
    if triole_position == 0:
        st = flags[position]
    elif triole_position == 1:
        st = flags[position] + TRIOLE_POS_1
    elif triole_position == 2:
        st = flags[position] + TRIOLE_POS_2
    return st

MIN_DURATION_DENOMINATOR = 32
DURATION_STEPS = 64
POSITION_STEPS = 16
TICKS_PER_BEAT = 1024
TRIOLE_POS_1 = (TICKS_PER_BEAT/12).__round__()
TRIOLE_POS_2 = (TICKS_PER_BEAT/6).__round__()
TICKS_PER_MIN_DURATION = TICKS_PER_BEAT*4/MIN_DURATION_DENOMINATOR
DURATION_BINS = np.arange(TICKS_PER_MIN_DURATION, (TICKS_PER_MIN_DURATION*DURATION_STEPS)+1, TICKS_PER_MIN_DURATION, dtype=int)

def write_midi(tokens, token2word, output_path):
    events = token_to_event(tokens, token2word)
    # get downbeat and note (no time)
    incorrect_notes = 0
    temp_notes = []
    for i in range(len(events)-4):
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
            if events[i+n+3]["name"] == 'Note-Duration' and events[i+n+3]["value"] == 'triole':
                index = int(events[i+n+2]["value"])-1
                duration = int(DURATION_BINS[index] / 3)
            else:
                index = int(events[i+n+2]["value"])-1
                duration = DURATION_BINS[index]
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
            flags = np.linspace(current_bar_st, current_bar_et, POSITION_STEPS, endpoint=False, dtype=int)
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
