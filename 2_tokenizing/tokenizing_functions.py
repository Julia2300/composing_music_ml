from os import walk
import miditoolkit
import numpy as np

MIN_DURATION_DENOMINATOR = 32
DURATION_STEPS = 64
POSITION_STEPS = 16
TICKS_PER_BEAT = 1024
TRIOLE_POS_1 = (TICKS_PER_BEAT/12).__round__()
TRIOLE_POS_2 = (TICKS_PER_BEAT/6).__round__()
TICKS_PER_MIN_DURATION = TICKS_PER_BEAT*4/MIN_DURATION_DENOMINATOR
DURATION_BINS = np.arange(TICKS_PER_MIN_DURATION, (TICKS_PER_MIN_DURATION*DURATION_STEPS)+1, TICKS_PER_MIN_DURATION, dtype=int)


def get_file_and_dirnames(p):
    f = []
    d = []
    for (dirpath, dirnames, filenames) in walk(p):
        f.extend(filenames)
        d.extend(dirnames)
        break
    return f,d

def split_into_max_len_items(item, duration, max_duration_in_bars, ticks_per_bar=TICKS_PER_BEAT*4):
    items = [] 
    full_lengths = duration // (max_duration_in_bars*ticks_per_bar)
    rest_ticks = duration % (max_duration_in_bars*ticks_per_bar)
    for i in range(full_lengths):
        items.append({
            "name": "Note",
            "start": item.start + i*max_duration_in_bars*ticks_per_bar,
            "end": item.start + (i+1)*max_duration_in_bars*ticks_per_bar,
            "pitch": item.pitch,
            "duration": max_duration_in_bars*ticks_per_bar
        })
    items.append({
            "name": "Note",
            "start": item.start + full_lengths*max_duration_in_bars*ticks_per_bar,
            "end": item.start + rest_ticks,
            "pitch": item.pitch,
            "duration": rest_ticks
        })
    return items

def convert_to_note_items(path, max_duration_in_bars, ticks_per_bar=TICKS_PER_BEAT*4):
    midi_obj = miditoolkit.midi.parser.MidiFile(path)
    notes = midi_obj.instruments[0].notes
    notes.sort(key=lambda x: (x.start, x.pitch))
    note_items = []
    for note in notes:
        duration = note.end - note.start
        if duration > max_duration_in_bars*ticks_per_bar:
            splitted_items = split_into_max_len_items(note, duration, max_duration_in_bars, ticks_per_bar)
            note_items.extend(splitted_items)
        else:
            note_items.append({
                "name": "Note",
                "start": note.start,
                "end": note.end,
                "pitch": note.pitch,
                "duration": note.end - note.start
            })
    note_items.sort(key=lambda x: x["start"])
    return note_items


def compute_shifts(items, ticks_per_position=1024*4/POSITION_STEPS):
    #grid
    grids = np.arange(0, items[-1]["start"]+1, ticks_per_position, dtype=int)
    # process
    for item in items:
        index = np.argmin(abs(grids - item["start"]))
        shift = item["start"] - grids[index]
        if shift == TRIOLE_POS_1 or shift == -TRIOLE_POS_2:
            item["triole_shift"] = TRIOLE_POS_1
        elif shift == -TRIOLE_POS_1:
            item["triole_shift"] = TRIOLE_POS_2
        else:
            item["triole_shift"] = shift
    return items 

def group_items(items, max_time, ticks_per_bar=TICKS_PER_BEAT*4):
    items.sort(key=lambda x: x["start"])
    downbeats = np.arange(0, max_time+ticks_per_bar, ticks_per_bar)
    groups = []
    for db1, db2 in zip(downbeats[:-1], downbeats[1:]):
        insiders = []
        for item in items:
            if (item["start"] >= db1) and (item["start"] < db2):
                insiders.append(item)
        overall = [db1] + insiders + [db2]
        groups.append(overall)
    return groups

def item2event(groups, triole_tokens, duration_steps, ticks_per_min_duration = TICKS_PER_BEAT*4/32):
    events = []
    n_downbeat = 0
    duration_bins = np.arange(ticks_per_min_duration, (ticks_per_min_duration*duration_steps)+1, ticks_per_min_duration, dtype=int)
    for i in range(len(groups)):
        if 'Note' not in [item["name"] for item in groups[i][1:-1]]:
            continue
        bar_st, bar_et = groups[i][0], groups[i][-1]
        n_downbeat += 1
        events.append({
            "name": "Bar",
            "time": None,
            "value": None,
            "text": n_downbeat 
        })
        for item in groups[i][1:-1]:
            ###### position ######
            flags = np.linspace(bar_st, bar_et, POSITION_STEPS, endpoint=False)
            index = np.argmin(abs(flags-item["start"]))
            # with triole tokens
            if triole_tokens:
                if item["start"] < flags[index]:
                    index -= 1
                events.append({
                    "name": "Position",
                    "time": item["start"],
                    "value": f"{index+1}/{POSITION_STEPS}",
                    "text": flags[index]
                })
                # triole positions
                if item["triole_shift"] == TRIOLE_POS_1:
                    events.append({
                        "name": "Position-Triole",
                        "time": item["start"],
                        "value": "1",
                        "text": flags[index]+TRIOLE_POS_1
                    })
                if item["triole_shift"] == TRIOLE_POS_2:
                    events.append({
                        "name": "Position-Triole",
                        "time": item["start"],
                        "value": "2",
                        "text": flags[index]+TRIOLE_POS_2
                    })
            # without triole tokens quantized to nearest flag
            else:
                events.append({
                    "name": "Position",
                    "time": item["start"],
                    "value": f"{index+1}/{POSITION_STEPS}",
                    "text": flags[index]
                })
            ###### note ######
            if item["name"] == 'Note':
                # pitch
                events.append({
                    "name": "Note-On",
                    "time": item["start"],
                    "value": item["pitch"],
                    "text": item["pitch"]
                })
                # duration
                duration = item["duration"]
                # with triole tokens
                if triole_tokens:
                    shift_normal = min(abs(duration_bins-duration))
                    shift_triole = min(abs(duration_bins-duration*3))
                    if shift_normal <= shift_triole:
                        index = np.argmin(abs(duration_bins-duration))
                        events.append({
                            "name": "Note-Duration",
                            "time": item["start"],
                            "value": index+1,
                            "text": f"{duration}/{duration_bins[index]}"
                        })
                    else:
                        index = np.argmin(abs(duration_bins-duration*3))
                        events.append({
                            "name": "Note-Duration",
                            "time": item["start"],
                            "value": index+1,
                            "text": f"{duration*3}/{duration_bins[index]}"
                        })
                        events.append({
                            "name": "Note-Duration",
                            "time": item["start"],
                            "value": "triole",
                            "text": f"{duration}/{duration_bins[index]}"
                        })
                # without triole tokens        
                else:
                    index = np.argmin(abs(duration_bins-duration))
                    events.append({
                        "name": "Note-Duration",
                        "time": item["start"],
                        "value": index+1,
                        "text": f"{duration}/{duration_bins[index]}"
                    })
    return events

def extract_events(input_path, duration_steps, triole_tokens):
    max_len_in_bars = int(duration_steps/32)
    note_items = convert_to_note_items(input_path, max_len_in_bars)
    note_items_shifts = compute_shifts(note_items)
    max_time = note_items_shifts[-1]["end"]
    grouped_items = group_items(note_items_shifts, max_time)
    events = item2event(grouped_items, triole_tokens, duration_steps)
    return events

def transform_to_prompt(words, max_bars=2):
    prompt = []
    bar = 0
    for word in words:
        if bar <= max_bars:
            if word == "Bar_None":
                bar += 1
            prompt.append(word)
        else:
            break
    return prompt