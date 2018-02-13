import numpy as np
import editdistance as ed
from math import log

def get_random_subsequence(sequences, rdm=None):
    """
    generate a random subsequence from a list of alphabetic sequences
    """
    if rdm is None : rdm = np.random.RandomState()
    selected_sequence = sequences[rdm.randint(len(sequences))]
    if len(selected_sequence) == 1 : return selected_sequence
    random_offset = rdm.randint(len(selected_sequence))
    random_length = rdm.randint(1, len(selected_sequence))
    return selected_sequence[random_offset:random_offset+random_length]

def sliding_ed(sequence, shapelet):
    """
    compute the minimum edit distance between a shapelet and (each subsequence of) a sequence
    """
    if len(shapelet) > len(sequence) : return ed.eval(sequence, shapelet)
    return sorted([ed.eval(sequence[i:i+len(shapelet)], shapelet) for i in range(len(sequence)-len(shapelet)+1)])[0]

def evaluate_candidate(candidate, edit_distances, labels, label_entropy, missing='lr', missing_data_labels=None):
    """
    evaluate a subsequence candidate
    """
    #sort labels wrt the edit distance 
    sorted_distances, sorted_labels = zip(*[(e, l) for (e, l) in sorted(zip(edit_distances, labels))])
    print('\ncandidate:', candidate)
    print(list(zip(sorted_distances, sorted_labels)))
    #all sequences have the same distance from the candidate: cannot split
    if len(set(sorted_distances)) == 1 : return {'subseq':candidate,
                                                 'ig':-1,
                                                 'z':0,
                                                 'margin':0,
                                               }
    #get all possible splits based on a threshold on the edit distance...
    all_splits = get_all_splits(sorted_labels, sorted_distances)
    #...and compute the corresponding ig and margin
    if missing == 'lr':
        evaluations = \
        [{'subseq':candidate,
          'ig':max(label_entropy - get_split_entropy(missing_data_labels+list(s[0]), s[1]),
                   label_entropy - get_split_entropy(s[0], list(s[1])+missing_data_labels)),
           #which value will be assigned to missing data at transformation time
          'z': 0 if get_split_entropy(missing_data_labels+list(s[0]), s[1]) < \
                    get_split_entropy(s[0], list(s[1])+missing_data_labels) else max(edit_distances)+1,
                
          'margin':sorted_distances[len(s[0])] - sorted_distances[len(s[0])-1],#break ig ties
          'threshold':sorted_distances[len(s[0])],
          'index':i,#just to preserve the order in case of ig+margin ties
        } for i, s in enumerate(all_splits)]
        
    else:
        evaluations = [{'subseq':candidate,
                        'ig':label_entropy - get_split_entropy(*s),
                        'z': None,
                        'margin':sorted_distances[len(s[0])] - sorted_distances[len(s[0])-1],
                        'threshold':sorted_distances[len(s[0])],
                        'index':i,
                        } for i, s in enumerate(all_splits)]  
        
    for i,s in enumerate(all_splits):
        print('split:{} ig:{:.3f} margin:{}'.format(s, evaluations[i]['ig'], evaluations[i]['margin']))
    
    #return the split yielding the maximum gain (margin is used to break ties)
    best_evaluation = sorted(evaluations, key = lambda e : (-e['ig'], -e['margin'], e['index']))[0]
    print('best split:', best_evaluation)
    return best_evaluation

def entropy(labels):
    """
    compute the entropy of a label set
    """
    n = len(labels)
    pos = sum(labels)
    pos_ratio = pos/n
    neg_ratio = (n - pos)/n
    return -0.0 if not pos_ratio or not neg_ratio else - (pos_ratio*log(pos_ratio, 2) + neg_ratio*log(neg_ratio, 2))

def get_all_splits(sorted_labels, sorted_distances):
    """
    compute all the possible ways of separating a labeled set based on a threshold on the edit distance
    """
    return [(sorted_labels[:split_index+1], sorted_labels[split_index+1:]) for split_index in np.where(np.diff(sorted_distances))[0]]

def get_split_entropy(a, b):
    """
    compute the entropy of a split (two separated label sets)
    """
    tot_len = len(a) + len(b)
    return len(a)/tot_len * entropy(a) + len(b)/tot_len * entropy(b)
