from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
import itertools
import pyter


def bleu(ref, gen):
    """
    calculate pair wise bleu score. uses nltk implementation
    :param ref - a list of reference sentences
    :param gen - a list of candidate(generated) sentences
    :return bleu score(float)
    """
    ref_bleu = []
    gen_bleu = []
    for l in gen:
        gen_bleu.append(l.split())
    for i, l in enumerate(ref):
        ref_bleu.append([l.split()])
    cc = SmoothingFunction()
    score_bleu = corpus_bleu(ref_bleu, gen_bleu, weights=(0, 1, 0, 0), smoothing_function=cc.method4)
    return score_bleu

# rouge scores for a reference/generated sentence pair
# source google seq2seq source code.


# supporting function
def _split_into_words(sentences):
    """Splits multiple sentences into words and flattens the result"""
    return list(itertools.chain(*[_.split(" ") for _ in sentences]))


# supporting function
def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences."""
    assert len(sentences) > 0
    assert n > 0

    words = _split_into_words(sentences)
    return _get_ngrams(n, words)


# supporting function
def _get_ngrams(n, text):
    """
    Calcualtes n-grams.
    :param n -  which n-grams to calculate
    :param text - An array of tokens
    :return - A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def rouge_n(reference_sentences, evaluated_sentences, n=2):
    """
    Computes ROUGE-N of two text collections of sentences.
    Source: http://research.microsoft.com/en-us/um/people/cyl/download/
    papers/rouge-working-note-v1.3.1.pdf
    :param evaluated_sentences - The sentences that have been picked by the summarizer
    :param reference_sentences - The sentences from the referene set
    :param n -  Size of ngram.  Defaults to 2.
    :return recall rouge score(float)
    :raises ValueError - raises exception if a param has len <= 0
    """
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        raise ValueError("Collections must contain at least 1 sentence.")

    evaluated_ngrams = _get_word_ngrams(n, evaluated_sentences)
    reference_ngrams = _get_word_ngrams(n, reference_sentences)
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    # Gets the overlapping ngrams between evaluated and reference
    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    # Handle edge case. This isn't mathematically correct, but it's good enough
    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))

    # just returning recall count in rouge, useful for our purpose
    return recall



def ter(ref, gen):
    """
    Calculates and returns the TER score between the references sequence to the generated sequences.
    :param  - ref - reference sentences - in a list
    :param - gen - generated sentences - in a list
    :return: averaged TER score over all sentence pairs
    """
    if len(ref) == 1:
        total_score =  pyter.ter(gen[0].split(), ref[0].split())
    else:
        total_score = 0
        for i in range(len(gen)):
            total_score = total_score + pyter.ter(gen[i].split(), ref[i].split())
        total_score = total_score/len(gen)
    return total_score


'''
read from files - 
ref.txt : reference texts
gen.txt : generated texts (from model)
these files should be in the same directory
'''


def evaluation_metrics(ref, gen, n_for_rouge=2):
    """
    :param - ref - reference file path -> file containing the reference sentences on each line
    :param - gen - model generated file path -> containing corresponding generated sentences(to reference sentences) on each line

    A list containing [bleu, rouge, meteor, ter]
    """

    for i, l in enumerate(gen):
        gen[i] = l.strip()

    for i, l in enumerate(ref):
        ref[i] = l.strip()

    ter_score = ter(ref, gen)
    bleu_score = bleu(ref, gen)
    rouge_score = rouge_n(ref, gen, n=n_for_rouge)

    return [bleu_score, rouge_score, ter_score]

