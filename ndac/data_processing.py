import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical


def quantile_classify(metric, sequences, split_quantiles=0.5, drop_class=None):
    """This function creates a new dataframe containing the specified
    metric and sequence and computes a new column, 'class', based on
    the high and low cuts"""

    dataframe = pd.concat([metric, sequences], axis=1)
    dataframe['class'] = 0
    print(len(metric), "samples input.")
    # make a histogram of the data to show the locations of the cut points
    hist = plt.figure(figsize=(6, 4))
    plt.hist(dataframe.iloc[:, 0], bins=100)
    plt.grid(b=0) #turn off grid lines
    plt.xlabel('Protein Expression Level', fontsize = 16)
    plt.xticks(fontsize =12)
    plt.ylabel('Counts', fontsize = 16)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    # if a single number is supplied, make high and low classes
    if np.isscalar(split_quantiles):
        cut_val = metric.quantile(split_quantiles)
        plt.axvline(cut_val, color='k', linestyle='dashed', linewidth=3)
        # assign class 1 to upper portion of data
        dataframe.loc[dataframe.iloc[:, 0] > cut_val, ['class']] = 1
        counts = pd.value_counts(dataframe['class'])
        print(counts[1], "samples above high cut,", counts[0],
              "samples below low cut.")
        return dataframe, hist

    for idx, cut in enumerate(split_quantiles):
        # add dashed line for cut point on histogram plot
        high_val = metric.quantile(cut)
        plt.axvline(high_val, linestyle='dashed', linewidth=3, color='k')
        # skip first cut since class = 0 by default
        if idx == 0:
            continue

        low_val = metric.quantile(split_quantiles[idx - 1])
        dataframe.loc[(dataframe.iloc[:, 0] <= high_val)
                      & (dataframe.iloc[:, 0] > low_val),
                      ['class']] = idx
    # add one more class above the final quantile cut value
    cut_val = metric.quantile(split_quantiles[-1])
    dataframe.loc[dataframe.iloc[:, 0] > cut_val,
                  ['class']] = len(split_quantiles)
    # drop classes if they are supplied
    if drop_class:
        dataframe = dataframe[-dataframe['class'].isin(drop_class)]
        # reassign classes based on number of unique remaining classes
        for idx, old_class in enumerate(np.unique(dataframe['class'])):
            dataframe.loc[dataframe['class'] == old_class, 'class'] = idx

    # get and print sample count for each class
    counts = pd.value_counts(dataframe['class']).sort_index()
    for idx, count in counts.iteritems():
        print(f'{count} samples in class {idx}')
    print(f'{len(metric) - len(dataframe)} samples removed.')

    return dataframe, hist


def value_classify(metric, sequences, split_values, drop_class=None):
    """This function creates a new dataframe containing the specified
    metric and sequence and computes a new column, 'class', based on
    the high and low values specified for metric"""

    dataframe = pd.concat([metric, sequences], axis=1)
    dataframe['class'] = 0
    print(len(metric), "samples input.")
    # make a histogram of the data to show the locations of the cut points
    hist = plt.figure(figsize=(6, 4))
    plt.hist(dataframe.iloc[:, 0], bins=20, color = [220 / 255, 100 / 255, 20 / 255])
    plt.grid(b=0) #turn off grid lines
    plt.xlabel('Protein Solubility Class', fontsize = 16)
    plt.xticks(fontsize =12)
    plt.ylabel('Counts', fontsize = 16)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    # if a single number is supplied, make high and low classes
    if np.isscalar(split_values):
        plt.axvline(split_values, color='k',
                    linestyle='dashed', linewidth=3)
        # assign class 1 to upper portion of data
        dataframe.loc[dataframe.iloc[:, 0] > split_values, ['class']] = 1
        counts = pd.value_counts(dataframe['class'])
        print(counts[1], "samples above high cut,", counts[0],
              "samples below low cut.")
        return dataframe, hist

    for idx, cut in enumerate(split_values):
        # add dashed line for cut point on histogram plot
        high_val = cut
        plt.axvline(high_val, linestyle='dashed', linewidth=3, color='k')
        # skip first cut since class = 0 by default
        if idx == 0:
            continue

        low_val = split_values[idx - 1]
        dataframe.loc[(dataframe.iloc[:, 0] <= high_val)
                      & (dataframe.iloc[:, 0] > low_val),
                      ['class']] = idx
    # add one more class above the final quantile cut value
    cut_val = split_values[-1]
    dataframe.loc[dataframe.iloc[:, 0] > cut_val,
                  ['class']] = len(split_values)
    # drop classes if they are supplied
    if drop_class:
        dataframe = dataframe[-dataframe['class'].isin(drop_class)]
        # reassign classes based on number of unique remaining classes
        for idx, old_class in enumerate(np.unique(dataframe['class'])):
            dataframe.loc[dataframe['class'] == old_class, 'class'] = idx

    # get and print sample count for each class
    counts = pd.value_counts(dataframe['class']).sort_index()
    for idx, count in counts.iteritems():
        print(f'{count} samples in class {idx}')
    print(f'{len(metric) - len(dataframe)} samples removed.')

    return dataframe, hist


def documentize_sequence(seqs, tag, nt_seq):
    """This function converts raw nucleotide or amino acid sequences
    into documents that can be encoded with the Keras Tokenizer().
    If a purification tag is supplied, it will be removed from
    the sequence document. """

    # setup 'docs' for use with Tokenizer
    def seq_to_doc(seq):
        # drop initial tag if it is supplied
        if tag:
            if tag not in seq:
                return None
            seq = seq.split(tag)[1]
        # split the sequence at every letter if not a nt_seq
        if not nt_seq:
            return ' '.join([aa for aa in seq])
        # split the sequence every 3 letters if it is a nt_seq
        return ' '.join([seq[i:i + 3]
                         for i in range(0, len(seq), 3)])

    return seqs.apply(seq_to_doc)


def encode_sequence(sequences, classes, max_length=0,
                    tag=None, nt_seq=True):
    dataframe = pd.concat([classes, sequences], axis=1)
    # if there are more than 4 letters, we should treat as amino acid seq
    if len(list(set(sequences[sequences.index[0]]))) > 4:
        nt_seq = False
    # turn sequence into documents for keras.Tokenizer()
    dataframe['seq_doc'] = documentize_sequence(sequences, tag, nt_seq)
    # drop sequences that weren't divisible by 3
    dataframe = dataframe[pd.notnull(dataframe['seq_doc'])]
    # define sequence documents
    docs = list(dataframe['seq_doc'])
    # create the tokenizer
    t = Tokenizer()
    # fit the tokenizer on the documents
    t.fit_on_texts(docs)
    # integer encode documents
    x = t.texts_to_sequences(docs)
    # pad or truncate sequences if max_length is nonzero
    if max_length:
        x = sequence.pad_sequences(x, maxlen=max_length)
    # get y value array (matrix for multiclass)
    y = dataframe.loc[:, dataframe.columns[0]].values
    if len(np.unique(y)) > 2:
        y = to_categorical(y)

    return x, y
