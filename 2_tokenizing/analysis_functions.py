import numpy as np

def check_duration(items):
    """
    Calculate the duration of each item by subtracting its start time from its end time.

    :param items: list of items, each item is a dictionary with 'start' and 'end' keys
    :return: list of durations of each item
    """
    durations = []
    for item in items:
        duration = item["end"] - item["start"]
        durations.append(duration)
    return durations

def get_duration_count(durations):
    """
    Count the frequency of each unique duration.

    :param durations: list of durations
    :return: dictionary where keys are unique durations and values are their frequencies
    """
    durations_count = {}
    for duration in set(durations):
        durations_count[duration] = durations.count(duration)
    return durations_count

def get_durations_in_bins(duration_dic, space):
    """
    Bin the durations into specified bins (space) and sum the counts for durations in each bin.

    :param duration_dic: dictionary where keys are durations and values are their frequencies
    :param space: list of bin edges
    :return: dictionary where keys are bin edges and values are the sum of counts for durations in each bin
    """
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
    """
    Check if notes in items form a triplet sequence of a specified duration and count sequences length.

    :param items: list of items, each item is a dictionary with 'start' and 'end' keys
    :param lower_bound: lower bound of the duration for an item to be considered as a triplet
    :param upper_bound: upper bound of the duration for an item to be considered as a triplet
    :return: list of sequence lengths of consecutive triplet notes
    """
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
    """
    Count the frequency of triplet sequence lengths.

    :param sequences: list of sequence lengths of consecutive triplet notes
    :return: dictionary where keys are unique lengths and values are their frequencies
    """
    trioles_count = {}
    for seq in set(sequences):
        trioles_count[seq] = sequences.count(seq)
    return trioles_count

def compute_shifts(items, ticks_per_position=1024*4/16):
    """
    Compute the shifts from each item's start time to the nearest grid position.

    :param items: list of items, each item is a dictionary with 'start' and 'end' keys
    :param ticks_per_position: number of ticks per grid position
    :return: list of shifts for each item
    """
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
    """
    Count the frequency of each shift.

    :param shifts: list of shifts for each item
    :return: dictionary where keys are shifts and values are their frequencies
    """
    shifts_count = {}
    for shift in set(shifts):
        shifts_count[shift] = shifts.count(shift)
    return shifts_count