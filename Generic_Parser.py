"""
Program to provide generic parsing for all files in user-specified directory.
The program assumes the input files have been scrubbed,
  i.e., HTML, ASCII-encoded binary, and any other embedded document structures that are not
  intended to be analyzed have been deleted from the file.

Dependencies:
    Python:  Load_MasterDictionary.py
    Data:    LoughranMcDonald_MasterDictionary_2014.csv

The program outputs:
   1.  File name
   2.  File size (in bytes)
   3.  Number of words (based on LM_MasterDictionary
   4.  Proportion of positive words (use with care - see LM, JAR 2016)
   5.  Proportion of negative words
   6.  Proportion of uncertainty words
   7.  Proportion of litigious words
   8.  Proportion of modal-weak words
   9.  Proportion of modal-moderate words
  10.  Proportion of modal-strong words
  11.  Proportion of constraining words (see Bodnaruk, Loughran and McDonald, JFQA 2015)
  12.  Number of alphanumeric characters (a-z, A-Z, 0-9)
  13.  Number of alphabetic characters (a-z, A-Z)
  14.  Number of digits (0-9)
  15.  Number of numbers (collections of digits)
  16.  Average number of syllables
  17.  Averageg word length
  18.  Vocabulary (see Loughran-McDonald, JF, 2015)

  ND-SRAF
  McDonald 2016/06
"""

import csv
import glob
import re
import string
import sys
import time
from nltk import ngrams
sys.path.append('D:\GD\Python\TextualAnalysis\Modules')  # Modify to identify path for custom modules
import Load_MasterDictionary as LM

# User defined directory for files to be parsed
TARGET_FILES = r'D:/Temp/TestParse/*.*'
# User defined file pointer to LM dictionary
MASTER_DICTIONARY_FILE = r'D:/GD/Research/Natural_Language_Processing/Dictionaries/' + \
                         'Master/LoughranMcDonald_MasterDictionary_2014.csv'
# User defined output file
OUTPUT_FILE = r'D:/Temp/Parser.csv'
# Setup output
OUTPUT_FIELDS = ['file name,', 'file size,', 'number of words,', '% positive,', '% negative,',
                 '% uncertainty,', '% litigious,', '% modal-weak,', '% modal moderate,',
                 '% modal strong,', '% constraining,', '# of alphanumeric,', '# of digits,',
                 '# of numbers,', 'avg # of syllables per word,', 'average word length,', 'vocabulary']

lm_dictionary = LM.load_masterdictionary(MASTER_DICTIONARY_FILE, True)


def get_dictonary_per_ngram_len(lm_dictionary):
    entries = lm_dictionary.values()
    lm_dictionary_max_ngram_size = max(map(lambda x: x.ngram_size, entries))
    # print(lm_dictionary_max_ngram_size)

    _lm_dictionary_per_ngram_len = {}

    for i in range(1, lm_dictionary_max_ngram_size + 1):
        _lm_dictionary_per_ngram_len[i] = {}

    for entry in entries:
        # print(entry.word)
        _lm_dictionary_per_ngram_len[entry.ngram_size][entry.word] = entry

    return _lm_dictionary_per_ngram_len


lm_dictionary_per_ngram_len = get_dictonary_per_ngram_len(lm_dictionary)
lm_dictionary_max_ngram_len = max(lm_dictionary_per_ngram_len.keys())
# print(lm_dictionary_max_ngram_len)


def main():

    f_out = open(OUTPUT_FILE, 'w')
    wr = csv.writer(f_out, lineterminator='\n')
    wr.writerow(OUTPUT_FIELDS)

    file_list = glob.glob(TARGET_FILES)
    for file in file_list:
        # print(file)
        with open(file, 'r', encoding='UTF-8', errors='ignore') as f_in:
            doc = f_in.read()
        doc_len = len(doc)
        # print('>' + doc + '<')
        doc = re.sub('(May|MAY)', ' ', doc)  # drop all May month references
        # print('>' + doc + '<')
        doc = doc.upper()  # for this parse caps aren't informative so shift

        output_data = get_data(doc)
        output_data[0] = file
        output_data[1] = doc_len
        wr.writerow(output_data)


def get_data(doc):

    vdictionary = {}
    _odata = [0] * 17
    total_syllables = 0
    word_length = 0

    clean_doc = doc = re.sub('[^\w]+', ' ', doc).strip()  # normalize whitespace and hyphens
    # print('>' + clean_doc + '<')
    clean_doc_tokens = clean_doc.split()

    # Create data structure to store occurence counts
    ngram_stats = {}
    for i in range(1, lm_dictionary_max_ngram_len + 1):
        ngram_stats[i] = {}
    for ngram_len in lm_dictionary_per_ngram_len:
        for ngram in lm_dictionary_per_ngram_len[ngram_len]:
            ngram_stats[ngram_len][ngram] = 0

    # Run through different ngram lengths and for each
    # split text into ngrams of that length, iterate over those
    # and increase occurence count for each one corresponding
    # to a dictionary entry
    for i in range(1, lm_dictionary_max_ngram_len + 1):
        doc_ngrams = ngrams(clean_doc_tokens, i)
        for doc_ngram in doc_ngrams:
            str_doc_ngram = ' '.join(doc_ngram)
            # print(str_doc_ngram)
            if str_doc_ngram in ngram_stats[i]:
                ngram_stats[i][str_doc_ngram] += 1

    # print()
    # print('Stats before correction:')
    # print('------------------------')
    # for i in range(1, lm_dictionary_max_ngram_len + 1):
    #     for ngram_n in lm_dictionary_per_ngram_len[i]:
    #         print(ngram_n + ': ' + str(ngram_stats[i][ngram_n]))

    # Correct counts for ngrams of lower order contained in ngrams of higher order
    # Note: This doesn't work reliably for overlapping ngrams. That would require
    # detailed tracking of the positiong of each occureance
    for i in range(1, lm_dictionary_max_ngram_len):
        for ngram_n in lm_dictionary_per_ngram_len[i]:
            for ngram_n_plus_1 in lm_dictionary_per_ngram_len[i + 1]:
                if ' ' + ngram_n + ' ' in ' ' + ngram_n_plus_1 + ' ':
                    ngram_stats[i][ngram_n] -= ngram_stats[i + 1][ngram_n_plus_1]

    # print()
    # print('Stats after correction:')
    # print('-----------------------')
    # for i in range(1, lm_dictionary_max_ngram_len + 1):
    #     for ngram_n in lm_dictionary_per_ngram_len[i]:
    #         print(ngram_n + ': ' + str(ngram_stats[i][ngram_n]))

    # For each ngram in the dictionary, accumulate the statistics based on 
    # the number of occurences of it
    for ngram_len in lm_dictionary_per_ngram_len:
        for ngram in lm_dictionary_per_ngram_len[ngram_len]:
            count = ngram_stats[ngram_len][ngram]
            if count > 0:
                if lm_dictionary[ngram].positive: _odata[3] += count
                if lm_dictionary[ngram].negative: _odata[4] += count
                if lm_dictionary[ngram].uncertainty: _odata[5] += count
                if lm_dictionary[ngram].litigious: _odata[6] += count
                if lm_dictionary[ngram].weak_modal: _odata[7] += count
                if lm_dictionary[ngram].moderate_modal: _odata[8] += count
                if lm_dictionary[ngram].strong_modal: _odata[9] += count
                if lm_dictionary[ngram].constraining: _odata[10] += count
                total_syllables += lm_dictionary[ngram].syllables * count

    # Leave the old logic here for counting words, average word length
    # and average syllables
    tokens = re.findall('\w+', doc)  # Note that \w+ splits hyphenated words
    for token in tokens:
        if not token.isdigit() and len(token) > 1 and token in lm_dictionary:
            _odata[2] += 1  # word count
            word_length += len(token)
            if token not in vdictionary:
                vdictionary[token] = 1

    _odata[11] = len(re.findall('[A-Z]', doc))
    _odata[12] = len(re.findall('[0-9]', doc))
    # drop punctuation within numbers for number count
    doc = re.sub('(?!=[0-9])(\.|,)(?=[0-9])', '', doc)
    doc = doc.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    _odata[13] = len(re.findall(r'\b[-+\(]?[$€£]?[-+(]?\d+\)?\b', doc))
    _odata[14] = total_syllables / _odata[2]
    _odata[15] = word_length / _odata[2]
    _odata[16] = len(vdictionary)
    
    # Convert counts to %
    for i in range(3, 10 + 1):
        _odata[i] = (_odata[i] / _odata[2]) * 100
    # Vocabulary
        
    return _odata


if __name__ == '__main__':
    print('\n' + time.strftime('%c') + '\nGeneric_Parser.py\n')
    main()
    print('\n' + time.strftime('%c') + '\nNormal termination.')
