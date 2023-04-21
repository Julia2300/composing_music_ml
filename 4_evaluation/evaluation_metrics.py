import numpy as np
from os import walk

def get_file_and_dirnames(p):
    f = []
    d = []
    for (dirpath, dirnames, filenames) in walk(p):
        f.extend(filenames)
        d.extend(dirnames)
        break
    return f,d

def get_tokens(token_sequence, start_token, end_token):
    tokens = []
    for token in token_sequence:
        if token in range(start_token, end_token+1):
            tokens.append(token)
    return tokens

def get_tokens_bar(token_sequence, bar_token, start_token, end_token):
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
    return len(list(set(pitches)))

def pitch_count_bar(bar_pitches):
    pitches = [list(set(bar)) for bar in bar_pitches]
    pitch_counts = [len(bar_pitches) for bar_pitches in pitches]
    return pitch_counts

def pitch_range_seq(pitches):
    return max(pitches) - min(pitches)

def pitch_range_bar(pitches_bar):
    return [max(bar) - min(bar) for bar in pitches_bar if len(bar)>0]

def pitch_intervals(pitches):
    pitch_intervals = []
    for i in range(len(pitches)-1):
        pitch_intervals.append(pitches[i+1]-pitches[i])
    return pitch_intervals

def pitch_class_histogram(pitches):
    pitch_classes = [(pitch-1) % 12 for pitch in pitches]
    pitch_classes = np.bincount(pitch_classes, minlength=12)
    return pitch_classes

def pitch_class_transition_matrix(pitches):
    transition_matrix = np.zeros([12,12], dtype=int)
    pitch_classes = [(pitch-1) % 12 for pitch in pitches]
    for i in range(len(pitch_classes)-1):
        transition_matrix[pitch_classes[i], pitch_classes[i+1]] += 1
    return transition_matrix


##### position based metrics #####
def note_count_seq(pitches):
    return len(pitches)

def note_count_bar(pitches_bar):
    note_counts = [len(bar) for bar in pitches_bar]
    return note_counts

def inter_onset_intervals(positions_bar):
    onset_intervals = []
    for i in range(len(positions_bar)):
        positions = positions_bar[i]
        for j in range(len(positions)-1):
            onset_intervals.append(positions[j+1]-positions[j])
        if i < len(positions_bar)-1:
            onset_intervals.append(positions_bar[i+1][0]-positions[-1]+16)
    return onset_intervals

def note_length_histogram(durations, start_duration_token):
    note_lengths = [duration-start_duration_token for duration in durations]
    length_classes = np.bincount(note_lengths, minlength=64)
    return length_classes

def note_length_transition_matrix(durations, start_duration_token, duration_bin):
    transition_matrix = np.zeros([duration_bin,duration_bin], dtype=int)
    note_lengths = [duration-start_duration_token for duration in durations]
    for i in range(len(note_lengths)-1):
        transition_matrix[note_lengths[i], note_lengths[i+1]] += 1
    return transition_matrix


###### get all metrics together ######
def get_metrics_for_multiple_sequences(token_data, token_flags):

    pitch_start_token = token_flags["start_pitch_token"]
    pitch_end_token = token_flags["end_pitch_token"]
    start_duration_token = token_flags["start_duration_token"]
    end_duration_token = token_flags["end_duration_token"]
    start_position_token = token_flags["start_position_token"]
    end_position_token = token_flags["end_position_token"]
    duration_bins = end_duration_token-start_duration_token+1

    evaluation_metrics = {
        "pitch_count_seq": [],
        "pitch_count_bar": [],
        "pitch_range_seq": [],
        "pitch_range_bar": [],
        "pitch_intervals_avg": [],
        "pitch_intervals_avg_abs": [],
        "pitch_intervals_hist": [],
        "pitch_class_hist_numbers": [],
        "pitch_class_hist_pitches": [],
        "pitch_class_transition_matrix": [],
        "note_count_seq": [],
        "note_count_bar": [],
        "onset_intervals_avg": [],
        "onset_intervals_hist": [],
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
        evaluation_metrics["pitch_count_seq"].append(pitch_count_seq(pitches))
        pitch_counts_bar = pitch_count_bar(pitches_bar)
        evaluation_metrics["pitch_count_bar"].append(np.round(np.mean(pitch_counts_bar), 2))

        evaluation_metrics["pitch_range_seq"].append(pitch_range_seq(pitches))
        pitch_ranges_bar = pitch_range_bar(pitches_bar)
        evaluation_metrics["pitch_range_bar"].append(np.round(np.mean(pitch_ranges_bar), 2))

        intervals = pitch_intervals(pitches)
        evaluation_metrics["pitch_intervals_avg"].append(np.round(np.mean(intervals), 2))
        evaluation_metrics["pitch_intervals_avg_abs"].append(np.round(np.mean(np.abs(intervals)), 2))
        evaluation_metrics["pitch_intervals_hist"].append(np.bincount(np.abs(intervals), minlength=36))

        pitch_classes = pitch_class_histogram(pitches)
        pitch_class_count = {}
        for i, key in enumerate(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]):
            pitch_class_count[key] = pitch_classes[i]
        evaluation_metrics["pitch_class_hist_numbers"].append(pitch_classes)
        evaluation_metrics["pitch_class_hist_pitches"].append(pitch_class_count)

        evaluation_metrics["pitch_class_transition_matrix"].append(pitch_class_transition_matrix(pitches))

        # rythm based metrics
        evaluation_metrics["note_count_seq"].append(note_count_seq(pitches))
        note_counts_bar = note_count_bar(pitches_bar)
        evaluation_metrics["note_count_bar"].append(np.round(np.mean(note_counts_bar), 2))

        onset_intervals = inter_onset_intervals(positions_bar)
        evaluation_metrics["onset_intervals_avg"].append(np.round(np.mean(onset_intervals), 2))
        evaluation_metrics["onset_intervals_hist"].append(np.bincount(np.abs(onset_intervals), minlength=32))

        note_lengths = note_length_histogram(durations, start_duration_token)
        note_lengths_count = {}
        for i in range(duration_bins+1):
            note_lengths_count[i] = note_lengths[i-1]
        evaluation_metrics["note_length_hist_numbers"].append(note_lengths)
        evaluation_metrics["note_length_hist_bins"].append(note_lengths_count)

        evaluation_metrics["note_length_transition_matrix"].append(note_length_transition_matrix(durations, start_duration_token, duration_bins))

    return evaluation_metrics


def summarize_evaluation_sequences(evaluation_metrics):
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

