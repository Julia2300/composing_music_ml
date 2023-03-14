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

def split_into_two_bar_items(item, duration, ticks_per_bar=TICKS_PER_BEAT*4):
    items = [] 
    two_bars = duration // (2*ticks_per_bar)
    rest_ticks = duration % (2*ticks_per_bar)
    for i in range(two_bars):
        items.append({
            "name": "Note",
            "start": item.start + i*2*ticks_per_bar,
            "end": item.start + (i+1)*2*ticks_per_bar,
            "pitch": item.pitch,
            "duration": 2*ticks_per_bar
        })
    items.append({
            "name": "Note",
            "start": item.start + two_bars*2*ticks_per_bar,
            "end": item.start + rest_ticks,
            "pitch": item.pitch,
            "duration": rest_ticks
        })
    return items

def convert_to_note_items(path, ticks_per_bar=TICKS_PER_BEAT*4):
    midi_obj = miditoolkit.midi.parser.MidiFile(path)
    notes = midi_obj.instruments[0].notes
    notes.sort(key=lambda x: (x.start, x.pitch))
    note_items = []
    for note in notes:
        duration = note.end - note.start
        if duration > 2*ticks_per_bar:
            split_into_two_bar_items(note, duration, ticks_per_bar)
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

def item2event(groups):
    events = []
    n_downbeat = 0
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

            # position
            flags = np.linspace(bar_st, bar_et, POSITION_STEPS, endpoint=False)
            index = np.argmin(abs(flags-item["start"]))
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
            # note tokens
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
                shift_normal = min(abs(DURATION_BINS-duration))
                shift_triole = min(abs(DURATION_BINS-duration*3))
                if shift_normal <= shift_triole:
                    index = np.argmin(abs(DURATION_BINS-duration))
                    events.append({
                        "name": "Note-Duration",
                        "time": item["start"],
                        "value": index+1,
                        "text": f"{duration}/{DURATION_BINS[index]}"
                    })
                else:
                    index = np.argmin(abs(DURATION_BINS-duration*3))
                    events.append({
                        "name": "Note-Duration",
                        "time": item["start"],
                        "value": index+1,
                        "text": f"{duration*3}/{DURATION_BINS[index]}"
                    })
                    events.append({
                        "name": "Note-Duration",
                        "time": item["start"],
                        "value": "triole",
                        "text": f"{duration}/{DURATION_BINS[index]}"
                    })
    return events

def extract_events(input_path):
    note_items = convert_to_note_items(input_path)
    note_items_shifts = compute_shifts(note_items)
    max_time = note_items_shifts[-1]["end"]
    grouped_items = group_items(note_items_shifts, max_time)
    events = item2event(grouped_items)
    return events