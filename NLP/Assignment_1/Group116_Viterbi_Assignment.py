# ==========================================================
# NLP ASSIGNMENT 1
# VITERBI DECODING FOR SEQUENCE TAGGING
#
# Group No: 116
# Domain: Movie Reviews
#
# Dataset:
# NLTK IMDb Movie Reviews Corpus
#
# ==========================================================

import nltk
import pandas as pd
from collections import defaultdict
from nltk.corpus import movie_reviews
from nltk.tokenize import sent_tokenize, word_tokenize

# ==========================================================
# DOWNLOAD REQUIRED DATA
# ==========================================================

print("\nDownloading required NLTK resources...\n")

nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# ==========================================================
# TASK 1
# LOAD DATASET
# EXTRACT FIRST 10,000 SENTENCES
# SPLIT 80/20
# ==========================================================

print("=" * 70)
print("TASK 1 : DATASET LOADING")
print("=" * 70)

all_sentences = []

for fileid in movie_reviews.fileids():

    text = movie_reviews.raw(fileid)

    for sentence in sent_tokenize(text):

        tokens = word_tokenize(sentence)

        if len(tokens) > 0:
            all_sentences.append(tokens)

print(f"\nTotal sentences available in corpus : {len(all_sentences)}")

sentences = all_sentences[:10000]

print(f"Using first 10,000 sentences        : {len(sentences)}")

split_index = int(0.8 * len(sentences))

train_sentences = sentences[:split_index]
test_sentences = sentences[split_index:]

print(f"Training sentences (80%)           : {len(train_sentences)}")
print(f"Testing sentences (20%)            : {len(test_sentences)}")

# ==========================================================
# TASK 2
# POS TAGGING
# START / END TAGS
# ==========================================================

print("\n")
print("=" * 70)
print("TASK 2 : POS TAGGING")
print("=" * 70)

tagged_train = []

for sentence in train_sentences:
    tagged_train.append(nltk.pos_tag(sentence))

print("\nPOS tagging completed.")

print("\nSample Tagged Sentence:\n")
print(tagged_train[0])

START = "<START>"
END = "<END>"

transition_counts = defaultdict(int)
emission_counts = defaultdict(int)
tag_counts = defaultdict(int)

for sentence in tagged_train:

    prev_tag = START

    for word, tag in sentence:

        transition_counts[(prev_tag, tag)] += 1
        emission_counts[(tag, word.lower())] += 1
        tag_counts[tag] += 1

        prev_tag = tag

    transition_counts[(prev_tag, END)] += 1

# ==========================================================
# INITIAL PROBABILITIES
# ==========================================================

print("\nCalculating Initial Probabilities...")

initial_probabilities = {}

total_sentences = len(tagged_train)

for tag in tag_counts.keys():

    count = transition_counts.get((START, tag), 0)

    initial_probabilities[tag] = count / total_sentences

print("\nSample Initial Probabilities")

for tag, prob in list(initial_probabilities.items())[:10]:
    print(f"{tag:8s} : {prob:.6f}")

# ==========================================================
# TRANSITION PROBABILITIES
# ==========================================================

print("\nCalculating Transition Probabilities...")

transition_prob = {}

for (prev_tag, curr_tag), count in transition_counts.items():

    total = sum(
        c
        for (p, t), c in transition_counts.items()
        if p == prev_tag
    )

    transition_prob[(prev_tag, curr_tag)] = count / total

print("\nSample Transition Probabilities")

sample_count = 0

for key, value in transition_prob.items():

    print(f"{key} -> {value:.6f}")

    sample_count += 1

    if sample_count >= 10:
        break

# ==========================================================
# EMISSION PROBABILITIES
# ==========================================================

print("\nCalculating Emission Probabilities...")

emission_prob = {}

for (tag, word), count in emission_counts.items():

    emission_prob[(tag, word)] = count / tag_counts[tag]

print("\nSample Emission Probabilities")

sample_count = 0

for key, value in emission_prob.items():

    print(f"{key} -> {value:.6f}")

    sample_count += 1

    if sample_count >= 10:
        break

# ==========================================================
# TASK 3
# VITERBI FROM SCRATCH
# ==========================================================

print("\n")
print("=" * 70)
print("TASK 3 : VITERBI IMPLEMENTATION")
print("=" * 70)

tags = list(tag_counts.keys())

UNKNOWN_WORD_PROB = 1e-6


def get_emission_probability(tag, word):

    return emission_prob.get(
        (tag, word.lower()),
        UNKNOWN_WORD_PROB
    )


def viterbi(words):

    V = [{}]
    path = {}

    # Initialization

    for tag in tags:

        init_prob = initial_probabilities.get(tag, 1e-10)

        emit_prob = get_emission_probability(
            tag,
            words[0]
        )

        V[0][tag] = init_prob * emit_prob

        path[tag] = [tag]

    # Recursion

    for t in range(1, len(words)):

        V.append({})

        new_path = {}

        for curr_tag in tags:

            emit_prob = get_emission_probability(
                curr_tag,
                words[t]
            )

            best_prob = -1

            best_prev_tag = None

            for prev_tag in tags:

                trans_prob = transition_prob.get(
                    (prev_tag, curr_tag),
                    1e-10
                )

                probability = (
                    V[t - 1][prev_tag]
                    * trans_prob
                    * emit_prob
                )

                if probability > best_prob:

                    best_prob = probability
                    best_prev_tag = prev_tag

            V[t][curr_tag] = best_prob

            new_path[curr_tag] = (
                path[best_prev_tag] + [curr_tag]
            )

        path = new_path

    best_final_tag = max(
        V[-1],
        key=lambda tag: V[-1][tag]
    )

    return V, path[best_final_tag]

# ==========================================================
# TASK 4
# DOMAIN-SPECIFIC SENTENCE
# 5 TO 8 WORDS
# ==========================================================

print("\n")
print("=" * 70)
print("TASK 4 : DOMAIN SPECIFIC SENTENCE")
print("=" * 70)

sentence = [
    "The",
    "movie",
    "was",
    "absolutely",
    "fantastic",
    "and",
    "engaging"
]

print("\nInput Sentence:")
print(" ".join(sentence))

print("\nRunning Viterbi Decoder...\n")

viterbi_matrix, predicted_tags = viterbi(sentence)

# ==========================================================
# DISPLAY TAG SEQUENCE
# ==========================================================

print("=" * 70)
print("FINAL PREDICTED TAG SEQUENCE")
print("=" * 70)

for word, tag in zip(sentence, predicted_tags):

    print(f"{word:15s} --> {tag}")

# ==========================================================
# DISPLAY VITERBI MATRIX
# ==========================================================

print("\n")
print("=" * 70)
print("VITERBI MATRIX (SLICE)")
print("=" * 70)

matrix_df = pd.DataFrame(viterbi_matrix)

matrix_df.columns = sentence

print(matrix_df.head(10))

# ==========================================================
# TASK 5
# DYNAMIC PROGRAMMING EXPLANATION
# ==========================================================

print("\n")
print("=" * 70)
print("TASK 5 : EXPLANATION")
print("=" * 70)

print("""
DYNAMIC PROGRAMMING NATURE OF VITERBI
-------------------------------------

The Viterbi algorithm uses Dynamic Programming.

Instead of evaluating every possible POS tag sequence,
it stores the best probability of reaching each tag
at every word position.

These intermediate results are stored inside the
Viterbi Matrix.

Each state depends only on:

1. Best previous state
2. Transition probability
3. Emission probability

This avoids repeated computations and significantly
reduces execution time.
""")

print("""
WHY VITERBI IS PREFERRED OVER EXHAUSTIVE SEARCH
------------------------------------------------

Suppose:

Sentence Length (N) = 7
Number of POS Tags (T) = 45

Exhaustive Search Complexity:

T^N

45^7

= 373,669,453,125 possible tag sequences

This is computationally infeasible.

Viterbi Complexity:

O(N × T²)

= 7 × 45²

= 14,175 operations approximately

Therefore, Viterbi is dramatically faster while still
producing the most probable tag sequence.
""")

print("""
UNKNOWN WORD HANDLING
---------------------

Words not present in the training data are assigned
a very small emission probability (1e-6).

This prevents probability from becoming zero and
allows decoding of unseen words.
""")

# ==========================================================
# SUMMARY
# ==========================================================

print("\n")
print("=" * 70)
print("ASSIGNMENT SUMMARY")
print("=" * 70)

print(f"Total Sentences Used        : {len(sentences)}")
print(f"Training Sentences          : {len(train_sentences)}")
print(f"Testing Sentences           : {len(test_sentences)}")
print(f"Unique POS Tags             : {len(tags)}")
print(f"Transition Probabilities    : {len(transition_prob)}")
print(f"Emission Probabilities      : {len(emission_prob)}")

print("\nAssignment completed successfully.")