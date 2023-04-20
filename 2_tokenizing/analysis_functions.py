import numpy as np

# now check duration times
def check_duration(items):
    durations = []
    for item in items:
        duration = item["end"] - item["start"]
        durations.append(duration)
    return durations

def get_duration_count(durations):
    durations_count = {}
    for duration in set(durations):
        durations_count[duration] = durations.count(duration)
    return durations_count

def get_durations_in_bins(duration_dic, space):
    updated_duration_dic = {}
    for i in space:
        updated_duration_dic[i] = 0

    for duration in duration_dic:
        for i in range(len(space)-1):
            if duration >= space[i] and duration < space[i+1]:
                updated_duration_dic[space[i]] += duration_dic[duration]
                break
        if duration > space[-1]:
            updated_duration_dic[space[-1]] += duration_dic[duration]

    return updated_duration_dic

def check_triole_seq(items, lower_bound=85, upper_bound=86):
    seq = []
    triole_seq = 0
    for item in items:
        duration = item["end"] - item["start"]
        if duration >= lower_bound and duration <= upper_bound:
            triole_seq += 1
        else:
            seq.append(triole_seq)
            triole_seq = 0
    return seq

def get_triole_count(sequences):
    trioles_count = {}
    for seq in set(sequences):
        trioles_count[seq] = sequences.count(seq)
    return trioles_count

def compute_shifts(items, ticks_per_position=1024*4/16):
    # grid
    grids = np.arange(0, items[-1]["start"]+1, ticks_per_position, dtype=float)
    grids = grids.astype(int)
    # process
    shifts = []
    for item in items:
        index = np.argmin(abs(grids - item["start"]))
        shift = grids[index] - item["start"]
        shifts.append(shift)
    return shifts

def get_shifts_count(shifts):
    shifts_count = {}
    for shift in set(shifts):
        shifts_count[shift] = shifts.count(shift)
    return shifts_count