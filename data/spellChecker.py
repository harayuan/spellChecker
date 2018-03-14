import re
import numpy as np
import itertools
from collections import Counter
import string
import operator
import pandas as pd
import nltk

# create a dictionary 'ERRORS_DICT' of key=correct-word and value=misspelled-word
ERRORS_DICT = {}
letters = 'abcdefghijklmnopqrstuvwxyz'


# we use re package to filter out unnecessary characters, tokenize the words and convert it to lower case
def tokens(text):
    return re.findall('[a-z]+', text.lower())


def errors():
    with open('spell-errors.txt') as filestream:
        for line in filestream:
            value_list = []
            currentline = line.split(':')
            # key is the correct-word
            key = currentline[0].strip().lower()
            # if there is only one value
            if ',' not in currentline[1]:
                # if value is not a correctly-spelled word
                if currentline[1].strip().lower() != key:
                    value_list.append(tokens(currentline[1].strip().lower())[0])
            # else if there are multiple values
            else:
                values = currentline[1].split(',')
                for v in values:
                    if v.strip().lower() != key:
                        value_list.append(tokens(v.strip().lower())[0])
            ERRORS_DICT[key] = value_list


# initialize the confusion matrices
DEL = np.zeros((26, 26))
INS = np.zeros((26, 26))
SUB = np.zeros((26, 26))
TRANS = np.zeros((26, 26))

# initialize a set where letter/letter pair will be stored
# these letter/letter pair will be used when counting their frequency in ERRORS_DICT
twoLetters = set()

ERRORS_DICT2 = {}


def errors2():
    with open('count_1w.txt') as filestream:
        for line in filestream:
            currentline = line.split('\t')
            ERRORS_DICT2[currentline[0]] = int(currentline[1])


errors2()


def updateConfusionMatrices(correct, wrong):
    """
    correct == key
    wrong == value
    """
    splits = [(correct[:i], correct[i:]) for i in range(len(correct) + 1)]

    # for all correct words, create the four lists of misspelled words
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]

    # identify the index in which difference between the two words(correct and wrong) occur

    # check which list the wrong word is included in
    if wrong in deletes:
        y_id = findDiff(correct, wrong)
        y = correct[y_id]
        x_id = y_id - 1
        x = correct[x_id]
        DEL[letters.find(x)][letters.find(y)] += 1
        twoLetters.add(x+y)

    elif wrong in inserts:
        y_id = findDiff(correct, wrong)
        y = wrong[y_id]
        x_id = y_id - 1
        x = wrong[x_id]
        INS[letters.find(x)][letters.find(y)] += 1

    elif wrong in replaces:
        y_id = findDiff(correct, wrong)
        y = correct[y_id]
        x_id = y_id
        x = wrong[x_id]
        SUB[letters.find(x)][letters.find(y)] += 1

    elif wrong in transposes:
        y_id = findDiff(correct, wrong)
        y = wrong[y_id]
        x_id = y_id + 1
        x = wrong[x_id]
        TRANS[letters.find(x)][letters.find(y)] += 1
        twoLetters.add(x + y)


def findDiff(correct, wrong):
    indices = []
    if correct != wrong:
        if len(correct) == len(wrong):
            for i in range(len(correct)):
                if correct[i] != wrong[i]:
                    indices.append(i)
        elif len(correct) != len(wrong):
            if len(correct) > len(wrong):
                long = correct
                short = wrong
            else:
                long = wrong
                short = correct
            for i in range(len(short)):
                size = len(short)
                if short == wrong[:size]:
                    indices.append(size)
                else:
                    for j in range(len(short)):
                        if short[j] != long[j]:
                            indices.append(j)
    else:
        indices.append(-1)
    return indices[0]


def createConfusionMatrices():
    for key in ERRORS_DICT.keys():
        for value in ERRORS_DICT[key]:
            updateConfusionMatrices(key, value)


def getInitialCounter():
    counter = {}
    alphabets = set(string.ascii_lowercase)
    myset = alphabets.union(twoLetters)
    # iterate through the misspelled words LIST in ERRORS_DICT
    for valueList in ERRORS_DICT.values():
        # iterate through misspelled WORD in the LIST
        for word in valueList:
            # iterate through each LETTER of the WORD
            for c in word:
                # if c is in myset
                if c in myset:
                    # if the LETTER is already in the dictionary
                    if c in counter.keys():
                        counter[c] += 1
                    else:
                        counter[c] = 1

            # iterate through each TWOLETTER of the WORD
            for i in range(len(word)-1):
                c1 = word[i]
                c2 = word[i+1]
                letters = c1 + c2
                if letters in myset:
                    if letters in counter.keys():
                        counter[letters] += 1
                    else:
                        counter[letters] = 1
    return counter


def getCount(char):
    if char in counter.keys():
        return counter[char]
    else:
        for valueList in ERRORS_DICT.values():
            for value in valueList:
                if char in counter.keys():
                    counter[char] += value.count(char)
                else:
                    counter[char] = value.count(char)
        return counter[char]


def words(text): return re.findall(r'\w+', text.lower())
# def words(text): return re.findall('[a-z]+', text.lower())


WORDS = Counter(words(open('big.txt').read()))


def P(word, N=sum(WORDS.values())):
    "Probability of `word`."
    return WORDS[word] / N


def P2(word):
    return ERRORS_DICT2[word] / total_cnt


def edits1(word):
    "All edits that are one edit away from `word`."
#     letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def edits2(word):
    return {e2 for e1 in edits1(word) for e2 in edits1(e1)}


def edits3(word):
    return {e2 for e1 in edits2(word) for e2 in edits2(e1)}


def edits4(word):
    return {e2 for e1 in edits3(word) for e2 in edits3(e1)}


def FromEdit2(word):
    words = edits2(word)
    dic = {}
    for w in words:
        if w in ERRORS_DICT.keys():
            dic[w] = P(w)
    return dic


def FromEdit3(word):
    words = edits3(word)
    dic = {}
    for w in words:
        if w in ERRORS_DICT.keys():
            dic[w] = P(w)

    return dic


def FromEdit4(word):
    words = edits4(word)
    dic = {}
    for w in words:
        if w in ERRORS_DICT.keys():
            dic[w] = P(w)
    return dic


def getDelP(wrong):
    # Try deleting a letter from the wrong word to get a correct word
    splits = [(wrong[:i], wrong[i:]) for i in range(len(wrong) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    dic = {}
    for delete in deletes:
        # if the created word is a real word
        if delete in ERRORS_DICT.keys():
            y_id = findDiff(delete, wrong)
            x_id = y_id - 1
            y = wrong[y_id]
            x = wrong[x_id]

            num = INS[letters.find(x)][letters.find(y)]
            # divide it by the frequency of x
            den = getCount(x)
            dic[delete] = (num/den) * P(delete)
    return dic


def getInsP(wrong):
    # Try inserting a letter to the wrong word to get a correct word
    splits = [(wrong[:i], wrong[i:]) for i in range(len(wrong) + 1)]
    inserts = [L + c + R for L, R in splits for c in letters]
    dic = {}
    for insert in inserts:
        if insert in ERRORS_DICT.keys():
            # letter y has been deleted after x
            y_id = findDiff(insert, wrong)
            y = insert[y_id]
            x_id = y_id - 1
            x = insert[x_id]

            num = DEL[letters.find(x)][letters.find(y)]
            den = getCount(x+y)
            dic[insert] = (num/den) * P(insert)
    return dic


def getSubP(wrong):
    # Try substituting a letter in the wrong word to get a correct word
    splits = [(wrong[:i], wrong[i:]) for i in range(len(wrong) + 1)]
    subs = [L + c + R[1:] for L, R in splits if R for c in letters]
    dic = {}
    for sub in subs:
        if sub in ERRORS_DICT.keys():
            x_id = findDiff(sub, wrong)
            y_id = findDiff(sub, wrong)
            x = wrong[x_id]
            y = sub[y_id]
            num = SUB[letters.find(x)][letters.find(y)]
            den = getCount(y)
            dic[sub] = (num/den) * P(sub)
    return dic


def getTransP(wrong):
    # Try transposing two consecutive letters in the wrong word to get a correct word
    splits = [(wrong[:i], wrong[i:]) for i in range(len(wrong) + 1)]
    trans = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    dic = {}
    for tran in trans:
        if tran in ERRORS_DICT.keys():
            y_id = findDiff(tran, wrong)
            x_id = y_id + 1
            y = wrong[y_id]
            x = wrong[x_id]
            num = TRANS[letters.find(x)][letters.find(y)]
            den = getCount(x+y)
            dic[tran] = (num/den) * P(tran)
    return dic


# initialize the ERRORS_DICT from training set
errors()
# train confusion matrices
createConfusionMatrices()
# count the frequencies of letter/two letters in our training set
counter = getInitialCounter()

# Test code with inputs from CSV file
# read test input from csv file
spell_test = pd.read_csv('test.csv')

# save column names to col_names_test
col_names_test = spell_test.columns.tolist()

# remove rows which have NaN values
spell_test.dropna(how='any', inplace=True)

# drop the ID column
spell_test_words = spell_test.drop('ID', 1)

# spell_test_words.head()

# create a list of wrong words
wrong_list = []
for index, row in spell_test_words.iterrows():
    wrong_word = row['WRONG']
    wrong_list.append(wrong_word)


# iterate through the wrong list, get the correct word, append it to a list

corrected_list = []

for i in range(len(wrong_list)):
    input_word = wrong_list[i]
    print(input_word)

    # Check if input_word is inside ERRORS_DICT
    corrects = []
    for k in ERRORS_DICT:
        value_list = ERRORS_DICT[k]
        if input_word in value_list:
            corrects.append(k)
    # If input_word is in the dictionary only once
    if(len(corrects) == 1):
        result_word = corrects[0]
    # If input_word is not in the dictionary or there are multiples
    else:
        # If input_word itself is a correct word
        if (input_word in ERRORS_DICT.keys()):
            result_word = input_word
        # If not, predict from edit distance 1
        else:
            DelDict = getDelP(input_word)
            InsDict = getInsP(input_word)
            SubDict = getSubP(input_word)
            TranDict = getTransP(input_word)

            # combine all dictionaries into one
            CombDict = {**DelDict, **InsDict, **SubDict, **TranDict}

            # If edit distance 1 did not work try edit distance 2
            if bool(CombDict) == False:
                Edit2Dict = FromEdit2(input_word)
                if bool(Edit2Dict) == False:
                    Edit3Dict = FromEdit3(input_word)
                    if bool(Edit3Dict) == False:
                        result_word = 'NULL'
                    else:
                        result_word = max(Edit3Dict.items(), key=operator.itemgetter(1))[0]
                else:
                    result_word = max(Edit2Dict.items(), key=operator.itemgetter(1))[0]
            else:
                result_word = max(CombDict.items(), key=operator.itemgetter(1))[0]
    print("Result is: " + result_word)
    corrected_list.append(result_word)

np.savetxt("prediction.csv", corrected_list, fmt="%s", delimiter=",", newline='\n')
