# Spell Checker

A spell checker based on noisy channel and uses confusion matrix to calculate spelling correction likelihood. This program corrects spelling of a corrupted word.

## Data Description
Training Datasets:
* big.txt: This file consists of lot of text. You can use this file to collect information about words. Number of
occurrences of word.
* spell-errors.txt: The format of every line in this text is the correct word is specified followed by the misspelled words. For example, like “raining: rainning, raning”. Form the 4 confusion matrices from this data set.
* count_1w.txt & count_2w.txt: count_1w.txt has text in the format: <word1> <no. of occurrences> and count_2w.txt
has text in the format: <word1> <word2> <no.of occurrences>. You can make use of these datasets to create much
better models.

Test Datasets:
* test.csv consists of 504 incorrect/misspelled words.

## Built With

* [Python 3](https://developer.android.com/index.html)

## Acknowledgments

* [A Spelling Correction Program Based on a Noisy Channel Model](http://www.aclweb.org/anthology/C90-2036)
