import math
import re
import numpy as np
from collections import defaultdict

from typing import NamedTuple, Dict, NewType, List, Tuple


def tokenize(message):
    message = message.lower()
    # 半角スペースなどで区切れる
    all_words = re.findall("[a-z0-9']+", message)
    return set(all_words)


class WordCount(NamedTuple):
    is_spam: int
    is_not_spam: int


def count_words(training_set: Tuple[str, bool]):
    # 1要素目がspamの数、2要素目がspamでない数
    counts: Dict[str, WordCount] = defaultdict(lambda: WordCount(is_spam=0, is_not_spam=0))

    for message, is_spam in training_set:
        for word in tokenize(message):
            if is_spam:
                counts[word].is_spam += 1
            else:
                counts[word].is_not_spam += 1
    return counts


class WordProbabilities(NamedTuple):
    """wordと、スパムが出た時、出ない時の、単語が出る確率"""
    word: str
    p_word_in_is_spam: int
    p_word_in_is_not_spam: int


def word_probabilities(counts: Dict[str, WordCount], total_spams, total_non_spams, k=0.5):
    """word_countsを、単語、p(単語|spam)、p(単語|¬spam)にする"""
    return [WordProbabilities(
        word=w,
        p_word_in_is_spam=(k + word_count.is_spam) / (total_spams + 2 * k),
        p_word_in_is_not_spam=(k + word_count.is_not_spam) / (total_non_spams + 2 * k)
    ) for w, word_count in counts.items()]


def spam_probability(word_probs: List[WordProbabilities], message):
    message_words = tokenize(message)
    log_prob_if_spam = log_prob_if_not_spam = 0.0

    for wprob in word_probs:
        # message_words内にword_probsがある場合
        # スパム中（又はスパムでないメール中）に単語を含む確率の対数を足す
        if wprob.word in message_words:
            log_prob_if_spam += math.log(wprob.p_word_in_is_spam)
            log_prob_if_not_spam += math.log(wprob.p_word_in_is_not_spam)

        # メッセージに単語が現れなかった場合
        # スパム中（又はスパムでないメール中）に単語を含まない確率の対数を足す
        else:
            log_prob_if_spam += math.log(1 - wprob.p_word_in_is_spam)
            log_prob_if_not_spam += math.log(1 - wprob.p_word_in_is_not_spam)

    # スパムのとき（スパムでないとき）、message中の単語が出る確率
    prob_if_spam = math.exp(log_prob_if_spam)
    prob_if_not_spam = math.exp(log_prob_if_not_spam)

    # P(X|S) / P(X) = P(S|X)
    return prob_if_spam / (prob_if_spam + prob_if_not_spam)


class NaiveBayesClassifier:
    def __init__(self, k=0.6):
        self.k = k
        self.word_probs = []

    def train(self, training_set: Tuple[str, bool]):
        num_spams = len([is_spam for message, is_spam in training_set if is_spam])
        num_non_spams = len(training_set) - num_spams

        word_counts = count_words(training_set)
        self.word_probs = word_probabilities(word_counts, num_spams, num_non_spams, self.k)

    def classify(self, message: str):
        return spam_probability(self.word_probs, message)




if __name__ == '__main__':
    pass
