#from sklearn.model_selection import LeaveOneOut
import numpy as np
#import copy
#import sklearn
from tqdm import tqdm
from scipy import stats, integrate
from sklearn.model_selection import LeaveOneOut


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


def get_distances(set_1, set_2, metrics):
    num_samples_1 = len(set_1[metrics[0]])
    num_samples_2 = len(set_2[metrics[0]])

    loo = LeaveOneOut()
    set1_intra = np.zeros((num_samples_1, len(metrics), num_samples_1 - 1))
    set2_intra = np.zeros((num_samples_2, len(metrics), num_samples_2 - 1))
    sets_inter_1_to_2 = np.zeros((num_samples_1, len(metrics), num_samples_2))

    # calculate intra-set distances
    for i, metric in enumerate(metrics):
        for train_index, test_index in loo.split(np.arange(num_samples_1)):
            set1_test_sample = set_1[metric][test_index[0]]
            set1_train_samples = [set_1[metric][index] for index in train_index]
            set1_intra[test_index[0]][i] = c_dist(
                set1_test_sample, set1_train_samples)
            
        for train_index, test_index in loo.split(np.arange(num_samples_2)):
            set2_test_sample = set_2[metric][test_index[0]]
            set2_train_samples = [set_2[metric][index] for index in train_index]
            set2_intra[test_index[0]][i] = c_dist(
                set2_test_sample, set2_train_samples)
            
    # calculate inter-set distances
    for i, metric in enumerate(metrics):
        try:
            for train_index, test_index in loo.split(np.arange(num_samples_1)):
                sets_inter_1_to_2[test_index[0]][i] = c_dist(set_1[metric][test_index[0]], set_2[metric])
        except ValueError:
            print(f"ValueError for {metric}")

    plot_set1_intra = np.transpose(
        set1_intra, (1, 0, 2)).reshape(len(metrics), -1)
    plot_set2_intra = np.transpose(
        set2_intra, (1, 0, 2)).reshape(len(metrics), -1)
    plot_sets_inter = np.transpose(
        sets_inter_1_to_2, (1, 0, 2)).reshape(len(metrics), -1)
    
    return plot_set1_intra, plot_set2_intra, plot_sets_inter


def get_kl_oa_metrics(set_1, set_2, metrics, plot_set1_intra, plot_set2_intra, plot_sets_inter):
    output = {}
    for i in tqdm(range(len(metrics))):
        
        mean1 = np.mean(set_1[metrics[i]], axis=0).tolist()
        std1 = np.std(set_1[metrics[i]], axis=0).tolist()
        mean2 = np.mean(set_2[metrics[i]], axis=0).tolist()
        std2 = np.std(set_2[metrics[i]], axis=0).tolist()
        try:
            kl1 = kl_dist(plot_set1_intra[i], plot_sets_inter[i])
            ol1 = overlap_area(plot_set1_intra[i], plot_sets_inter[i])
            kl2 = kl_dist(plot_set2_intra[i], plot_sets_inter[i])
            ol2 = overlap_area(plot_set2_intra[i], plot_sets_inter[i])
        except:
            kl1, ol1, kl2, ol2 = "error", "error", "error", "error"

        output[metrics[i]] = {
            "mean_set1": mean1,
            "std_set1": std1,
            "mean_set2": mean2,
            "std_set2": std2,
            "kl_set1_inter": kl1,
            "ol_set1_inter": ol1,
            "kl_set2_inter": kl2,
            "ol_set2_inter": ol2
        }
        
    return output


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