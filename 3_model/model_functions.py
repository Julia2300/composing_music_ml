from torch.utils.data import Dataset
from transformers import GPT2Config, GPT2LMHeadModel, TrainingArguments, Trainer
import miditoolkit


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
