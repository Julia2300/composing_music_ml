import numpy as np
from os import walk

def get_file_and_dirnames(p):
    """
    Get filenames and directory names in a given directory.

    :param p: path of directory
    :return: list of filenames, list of directory names
    """
    f = []
    d = []
    for (dirpath, dirnames, filenames) in walk(p):
        f.extend(filenames)
        d.extend(dirnames)
        break
    return f,d

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

##### token filter functions #####

def get_tokens(token_sequence, start_token, end_token):
    """
    Get all tokens within a specified token id range from a sequence.

    :param token_sequence: sequence of tokens
    :param start_token: start of range
    :param end_token: end of range
    :return: list of tokens within specified range
    """
    tokens = []
    for token in token_sequence:
        if token in range(start_token, end_token+1):
            tokens.append(token)
    return tokens

def get_tokens_bar(token_sequence, bar_token, start_token, end_token):
    """
    Get all tokens within a specified token id range from a sequence, grouped by bars.

    :param token_sequence: sequence of tokens
    :param bar_token: bar token value
    :param start_token: start of range
    :param end_token: end of range
    :return: list of tokens within specified range, grouped by bars
    """
    tokens = []
    i = -1
    for token in token_sequence:
        if token == bar_token:
            i += 1
            tokens.append([])
        elif token in range(start_token, end_token+1):
            tokens[i].append(token)
    return tokens

##### pitch based metrics #####
def pitch_count_seq(pitches):
    """
    Get the count of unique pitches in a sequence.

    :param pitches: List of pitches in a sequence
    :return: Count of unique pitches
    """
    return len(list(set(pitches)))

def pitch_count_bar(bar_pitches):
    """
    Get the count of unique pitches in each bar.

    :param bar_pitches: List of lists, where each list represents a bar and contains pitches
    :return: List of unique pitch counts for each bar
    """
    pitches = [list(set(bar)) for bar in bar_pitches]
    pitch_counts = [len(bar_pitches) for bar_pitches in pitches]
    return pitch_counts

def pitch_range_seq(pitches):
    """
    Get the range of pitches in a sequence.

    :param pitches: List of pitches in a sequence
    :return: Range of pitches
    """
    return max(pitches) - min(pitches)

def pitch_range_bar(pitches_bar):
    """
    Get the range of pitches in each bar.

    :param pitches_bar: List of lists, where each list represents a bar and contains pitches
    :return: List of pitch ranges for each bar
    """
    return [max(bar) - min(bar) for bar in pitches_bar if len(bar)>0]

def pitch_intervals(pitches):
    """
    Calculate the intervals between successive pitches in a sequence.

    :param pitches: List of pitches in a sequence
    :return: List of pitch intervals
    """
    pitch_intervals = []
    for i in range(len(pitches)-1):
        pitch_intervals.append(pitches[i+1]-pitches[i])
    return pitch_intervals

def pitch_class_histogram(pitches):
    """
    Get the histogram of pitch classes in a sequence.

    :param pitches: List of pitches in a sequence
    :return: Histogram of pitch classes
    """
    pitch_classes = [(pitch-1) % 12 for pitch in pitches]
    pitch_classes = np.bincount(pitch_classes, minlength=12)
    return pitch_classes

def pitch_class_transition_matrix(pitches):
    """
    Get the transition matrix for pitch classes in a sequence.

    :param pitches: List of pitches in a sequence
    :return: Transition matrix for pitch classes
    """
    transition_matrix = np.zeros([12,12], dtype=int)
    pitch_classes = [(pitch-1) % 12 for pitch in pitches]
    for i in range(len(pitch_classes)-1):
        transition_matrix[pitch_classes[i], pitch_classes[i+1]] += 1
    return transition_matrix


##### position based metrics #####
def note_count_seq(pitches):
    """
    Get the count of notes in a sequence.

    :param pitches: List of pitches in a sequence
    :return: Count of notes
    """
    return len(pitches)

def note_count_bar(pitches_bar):
    """
    Get the count of notes in each bar.

    :param pitches_bar: List of lists, where each list represents a bar and contains pitches
    :return: List of note counts for each bar
    """
    note_counts = [len(bar) for bar in pitches_bar]
    return note_counts

def inter_onset_intervals(positions_bar):
    """
    Calculate the onset intervals in a sequence of positions.

    :param positions_bar: List of lists, where each list represents a bar and contains note positions
    :return: List of onset intervals
    """
    onset_intervals = []
    for i in range(len(positions_bar)):
        positions = positions_bar[i]
        for j in range(len(positions)-1):
            onset_intervals.append(positions[j+1]-positions[j])
        if i < len(positions_bar)-1 and len(positions_bar[i+1]) > 0 and len(positions) > 0:
            onset_intervals.append(positions_bar[i+1][0]-positions[-1]+16)
    return onset_intervals

def note_length_histogram(durations, start_duration_token):
    """
    Get the histogram of note lengths in a sequence.

    :param durations: List of durations in a sequence
    :param start_duration_token: Start duration token
    :return: Histogram of note lengths
    """
    note_lengths = [duration-start_duration_token for duration in durations]
    length_classes = np.bincount(note_lengths, minlength=64)
    return length_classes

def note_length_transition_matrix(durations, start_duration_token, duration_bin):
    """
    Get the transition matrix for note lengths in a sequence.

    :param durations: List of durations in a sequence
    :param start_duration_token: Start duration token
    :param duration_bin: Duration bins size
    :return: Transition matrix for note lengths
    """
    transition_matrix = np.zeros([duration_bin,duration_bin], dtype=int)
    note_lengths = [duration-start_duration_token for duration in durations]
    for i in range(len(note_lengths)-1):
        transition_matrix[note_lengths[i], note_lengths[i+1]] += 1
    return transition_matrix


###### get all metrics together ######
def get_metrics_for_multiple_sequences(token_data, token_flags):
    """
    Extract various music metrics for token sequences.

    :param token_data: list of token sequences
    :param token_flags: dictionary of tokens and their corresponding token ids
    :return: dictionary of calculated metrics
    """

    pitch_start_token = token_flags["start_pitch_token"]
    pitch_end_token = token_flags["end_pitch_token"]
    start_duration_token = token_flags["start_duration_token"]
    end_duration_token = token_flags["end_duration_token"]
    start_position_token = token_flags["start_position_token"]
    end_position_token = token_flags["end_position_token"]
    duration_bins = end_duration_token-start_duration_token+1
    pitch_range = pitch_end_token-pitch_start_token+1

    evaluation_metrics = {
        "mean_pitch": [],
        "pitch_count_seq": [],
        "pitch_count_bar": [],
        "pitch_range_seq": [],
        "pitch_range_bar": [],
        "pitch_intervals_avg": [],
        "pitch_intervals_avg_abs": [],
        "pitch_intervals_hist": [],
        "pitch_class_hist_numbers": [],
        "pitch_class_hist_pitches": [],
        "crooked_pitch_count": [],
        "pitch_class_transition_matrix": [],
        "note_count_seq": [],
        "note_count_bar": [],
        "onset_intervals_avg": [],
        "onset_intervals_hist": [],
        "mean_duration": [],
        "note_length_hist_numbers": [],
        "note_length_hist_bins": [],
        "note_length_transition_matrix": [],
    }

    for token_seq in token_data:
        pitches = get_tokens(token_seq, pitch_start_token, pitch_end_token)
        pitches_bar = get_tokens_bar(token_seq, 0, pitch_start_token, pitch_end_token)
        positions_bar = get_tokens_bar(token_seq, 0, start_position_token, end_position_token)
        durations = get_tokens(token_seq, start_duration_token, end_duration_token)
        
        # pitch based metrics
        evaluation_metrics["mean_pitch"].append(np.round(np.mean(pitches), 2))

        evaluation_metrics["pitch_count_seq"].append(pitch_count_seq(pitches))
        pitch_counts_bar = pitch_count_bar(pitches_bar)
        evaluation_metrics["pitch_count_bar"].append(np.round(np.mean(pitch_counts_bar), 2))

        evaluation_metrics["pitch_range_seq"].append(pitch_range_seq(pitches))
        pitch_ranges_bar = pitch_range_bar(pitches_bar)
        evaluation_metrics["pitch_range_bar"].append(np.round(np.mean(pitch_ranges_bar), 2))

        intervals = pitch_intervals(pitches)
        evaluation_metrics["pitch_intervals_avg"].append(np.round(np.mean(intervals), 2))
        evaluation_metrics["pitch_intervals_avg_abs"].append(np.round(np.mean(np.abs(intervals)), 2))
        evaluation_metrics["pitch_intervals_hist"].append(np.bincount(np.abs(intervals), minlength=pitch_range))

        pitch_classes = pitch_class_histogram(pitches)
        pitch_class_count = {}
        for i, key in enumerate(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]):
            pitch_class_count[key] = pitch_classes[i]
        evaluation_metrics["pitch_class_hist_numbers"].append(pitch_classes)
        evaluation_metrics["pitch_class_hist_pitches"].append(pitch_class_count)

        evaluation_metrics["pitch_class_transition_matrix"].append(pitch_class_transition_matrix(pitches))

        crooked_pitches = ["C#", "D#", "F#", "G#", "A#"]
        crooked_pitch_count = 0
        for pitch in crooked_pitches:
            crooked_pitch_count += pitch_class_count[pitch]
        evaluation_metrics["crooked_pitch_count"].append(crooked_pitch_count)

        # rythm based metrics
        evaluation_metrics["note_count_seq"].append(note_count_seq(pitches))
        note_counts_bar = note_count_bar(pitches_bar)
        evaluation_metrics["note_count_bar"].append(np.round(np.mean(note_counts_bar), 2))

        onset_intervals = inter_onset_intervals(positions_bar)
        evaluation_metrics["onset_intervals_avg"].append(np.round(np.mean(onset_intervals), 2))
        evaluation_metrics["onset_intervals_hist"].append(np.bincount(np.abs(onset_intervals), minlength=32))

        evaluation_metrics["mean_duration"].append(np.round(np.mean([dur-start_duration_token+1 for dur in durations]), 2))

        note_lengths = note_length_histogram(durations, start_duration_token)
        note_lengths_count = {}
        for i in range(duration_bins+1):
            note_lengths_count[i] = note_lengths[i-1]
        evaluation_metrics["note_length_hist_numbers"].append(note_lengths)
        evaluation_metrics["note_length_hist_bins"].append(note_lengths_count)

        evaluation_metrics["note_length_transition_matrix"].append(note_length_transition_matrix(durations, start_duration_token, duration_bins))

    return evaluation_metrics


def summarize_evaluation_sequences(evaluation_metrics):
    """
    Summarize evaluation metrics by taking the mean for each scalar metric and the sum for more-dimensional metrics across sequences.

    :param evaluation_metrics: dictionary of metrics
    :return: dictionary of averaged metrics
    """
    averaged_and_added_evaluation_metrics = {}
    for key in evaluation_metrics.keys():
        shape = np.array(evaluation_metrics[key]).shape
        if len(shape) == 1:
            if type(evaluation_metrics[key][0]) != dict:
                averaged_and_added_evaluation_metrics[key] = np.round(np.mean(evaluation_metrics[key]), 2)
            elif type(evaluation_metrics[key][0]) == dict:
                added_dict = {}
                for sub_key in evaluation_metrics[key][0].keys():
                    added_dict[sub_key] = np.sum([evaluation_metrics[key][i][sub_key] for i in range(len(evaluation_metrics[key]))])
                averaged_and_added_evaluation_metrics[key] = added_dict
        else:
            added_array = evaluation_metrics[key][0]
            for next_array in evaluation_metrics[key][1:]:
                added_array = np.add(added_array, next_array)
            averaged_and_added_evaluation_metrics[key] = added_array
    return averaged_and_added_evaluation_metrics

def std_evaluation_sequences(evaluation_metrics):
    """
    Calculate the standard deviation of each evaluation metric across sequences.

    :param evaluation_metrics: dictionary of metrics
    :return: dictionary of standard deviations
    """
    std_evaluation_metrics = {}
    for key in evaluation_metrics.keys():
        shape = np.array(evaluation_metrics[key]).shape
        if len(shape) == 1:
            if type(evaluation_metrics[key][0]) != dict:
                std_evaluation_metrics[key] = np.round(np.std(evaluation_metrics[key]), 2)
            elif type(evaluation_metrics[key][0]) == dict:
                std_evaluation_metrics[key] = "not mean"
        else:
            std_evaluation_metrics[key] = "not mean"
    return std_evaluation_metrics