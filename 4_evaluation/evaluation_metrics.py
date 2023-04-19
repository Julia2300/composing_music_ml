import numpy as np

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

#def get_positions(token_sequence, start_position_token, end_position_token):
#    positions = []
#    for token in token_sequence:
#        if token in range(start_position_token, end_position_token+1):
#            positions.append(token)
#    return positions

##### pitch based metrics #####
def pitch_count_seq(pitches):
    return len(list(set(pitches)))

def pitch_count_bar(bar_pitches):
    pitches = [list(set(bar)) for bar in bar_pitches]
    pitch_counts = [len(bar_pitches) for bar_pitches in pitches]
    return pitch_counts

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

def pitch_range_seq(pitches):
    return max(pitches) - min(pitches)

def pitch_range_bar(pitches_bar):
    return [max(bar) - min(bar) for bar in pitches_bar]

def pitch_intervals(pitches):
    pitch_intervals = []
    for i in range(len(pitches)-1):
        pitch_intervals.append(pitches[i+1]-pitches[i])
    return pitch_intervals


##### position based metrics #####
def note_count_seq(pitches):
    return len(pitches)

def note_count_bar(pitches_bar):
    note_counts = [len(bar) for bar in pitches_bar]
    return note_counts

