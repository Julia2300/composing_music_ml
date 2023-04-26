#from sklearn.model_selection import LeaveOneOut
import numpy as np
#import copy
#import sklearn
from tqdm import tqdm
from scipy import stats, integrate


###### distance functions ######
# inspired by Yang and Lerch

def c_dist(A, B):
    c_dist = np.zeros(len(B))
    for i in range(0, len(B)):
        c_dist[i] = np.linalg.norm(A - B[i])
    return c_dist

# Calculate overlap between the two PDF
def overlap_area(A, B):
    pdf_A = stats.gaussian_kde(A)
    pdf_B = stats.gaussian_kde(B)
    return integrate.quad(lambda x: min(pdf_A(x), pdf_B(x)), np.min((np.min(A), np.min(B))), np.max((np.max(A), np.max(B))))[0]


# Calculate KL distance between the two PDF
def kl_dist(A, B, num_sample=1000):
    pdf_A = stats.gaussian_kde(A)
    pdf_B = stats.gaussian_kde(B)
    sample_A = np.linspace(np.min(A), np.max(A), num_sample)
    sample_B = np.linspace(np.min(B), np.max(B), num_sample)
    return stats.entropy(pdf_A(sample_A), pdf_B(sample_B))



###### overlapping functions ######
def get_bars(song):
    bars = []
    start = 1
    for i in range(1, len(song)):
        if song[i] == 0:
            bars.append(song[start:i])
            start = i + 1
    bars.append(song[start:])
    return bars

def get_overlapping_sequences(song1, song2):
    overlapping_length = 0
    overlapping_lengths_list = []
    skips_i = []
    skips_j = []
    for i in range(len(song1)):
        if i in skips_i:
            continue
        for j in range(len(song2)):
            if j in skips_j:
                continue
            if song1[i] == song2[j]:
                counter_i = i
                counter_j = j
                while counter_i<len(song1) and counter_j<len(song2) and song1[counter_i] == song2[counter_j]:
                    overlapping_length += 1
                    skips_i.append(counter_i)
                    skips_j.append(counter_j)
                    counter_i += 1
                    counter_j += 1 
                overlapping_lengths_list.append(overlapping_length)
                overlapping_length = 0
    return overlapping_lengths_list

def get_overlap(token_data_1, token_data_2, bars=True):
    max_overlap = []
    overlap_count = []
    for i in tqdm(range(len(token_data_1))):
        for j in range(len(token_data_2)):
            if bars:
                song1 = get_bars(token_data_1[i])
                song2 = get_bars(token_data_2[j])
            else:
                song1 = token_data_1[i]
                song2 = token_data_2[j]
            overlapping_seq = get_overlapping_sequences(song1, song2)
            overlap_count.append(len(overlapping_seq))
            if len(overlapping_seq) > 0:
                max_overlap.append(max(overlapping_seq))
            else:
                max_overlap.append(0)
    return overlap_count, max_overlap