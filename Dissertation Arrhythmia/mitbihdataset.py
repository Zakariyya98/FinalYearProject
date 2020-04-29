import numpy as np
import flask as flask
import wfdb
from glob import glob

def get_data():
    # We want the data from the .atr records
    dataset = glob('./arrhythmiadb/*.atr')
    dataset = [path[:-4] for path in dataset]
    dataset.sort()

    return dataset

def heart_beat_classification(annotation):
   
    # 'N' Specifies a normal heartbeat
    goodbeat = ['N']   
    ids = np.in1d(annotation.symbol, goodbeat)
    heart_beat = annotation.sample[ids]
    return heart_beat
  
def segment_records(records):
    Normal_Beats = []
    for e in records:
        signals, fields = wfdb.rdsamp(e, channels = [0]) 

        ann = wfdb.rdann(e, 'atr')
        good = ['N']
        ids = np.in1d(ann.symbol, good)
        imp_beats = ann.sample[ids]
        beats = (ann.sample)
        for i in imp_beats:
            beats = list(beats)
            j = beats.index(i)
            if(j!=0 and j!=(len(beats)-1)):
                x = beats[j-1]
                y = beats[j+1]
                diff1 = abs(x - beats[j])//2
                diff2 = abs(y - beats[j])//2
                Normal_Beats.append(signals[beats[j] - diff1: beats[j] + diff2, 0])
    return Normal_Beats