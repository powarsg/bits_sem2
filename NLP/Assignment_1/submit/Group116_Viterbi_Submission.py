# ==========================================================
# NLP ASSIGNMENT 1
# Viterbi Decoding for Sequence Tagging
#
# Group No: 116
# Domain: Movie / Restaurant Reviews
# Dataset: NLTK IMDb Movie Reviews Corpus
# ==========================================================

import nltk
import pandas as pd
from collections import defaultdict
from nltk.corpus import movie_reviews
from nltk.tokenize import sent_tokenize, word_tokenize

START = "<START>"
END = "<END>"
UNKNOWN_WORD_PROB = 1e-6


def download_nltk_resources():
    """Download required NLTK corpora and taggers."""
    print("\nDownloading required NLTK resources...\n")
    for resource in (
        "punkt",
        "punkt_tab",
        "movie_reviews",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
    ):
        nltk.download(resource, quiet=True)


def load_and_split_dataset(num_sentences=10000, train_ratio=0.8):
    """
    Task 1: Load domain dataset, take first 10,000 sentences,
    split into 80% train / 20% test.
    """
    print("=" * 70)
    print("TASK 1 : DATASET LOADING")
    print("=" * 70)

    all_sentences = []
    for fileid in movie_reviews.fileids():
        text = movie_reviews.raw(fileid)
        for sentence in sent_tokenize(text):
            tokens = word_tokenize(sentence)
            if tokens:
                all_sentences.append(tokens)

    print(f"\nTotal sentences available in corpus : {len(all_sentences)}")
    sentences = all_sentences[:num_sentences]
    print(f"Using first {num_sentences:,} sentences        : {len(sentences)}")

    split_index = int(train_ratio * len(sentences))
    train_sentences = sentences[:split_index]
    test_sentences = sentences[split_index:]

    print(f"Training sentences ({int(train_ratio * 100)}%)           : {len(train_sentences)}")
    print(f"Testing sentences ({int((1 - train_ratio) * 100)}%)            : {len(test_sentences)}")

    return train_sentences, test_sentences


def tag_and_build_probabilities(train_sentences):
    """
    Task 2: POS-tag training data and compute emission / transition
    probabilities with <START> and <END> sentence boundaries.
    """
    print("\n")
    print("=" * 70)
    print("TASK 2 : POS TAGGING")
    print("=" * 70)

    tagged_train = [nltk.pos_tag(sentence) for sentence in train_sentences]
    print("\nPOS tagging completed.")
    print("\nSample Tagged Sentence:\n")
    print(tagged_train[0])

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

    total_sentences = len(tagged_train)
    initial_probabilities = {
        tag: transition_counts.get((START, tag), 0) / total_sentences
        for tag in tag_counts
    }

    transition_prob = {}
    for (prev_tag, curr_tag), count in transition_counts.items():
        total = sum(
            c for (p, _), c in transition_counts.items() if p == prev_tag
        )
        transition_prob[(prev_tag, curr_tag)] = count / total

    emission_prob = {
        (tag, word): count / tag_counts[tag]
        for (tag, word), count in emission_counts.items()
    }

    print("\nCalculating Initial Probabilities...")
    print("\nSample Initial Probabilities")
    for tag, prob in list(initial_probabilities.items())[:10]:
        print(f"{tag:8s} : {prob:.6f}")

    print("\nCalculating Transition Probabilities...")
    print("\nSample Transition Probabilities")
    for i, (key, value) in enumerate(transition_prob.items()):
        print(f"{key} -> {value:.6f}")
        if i >= 9:
            break

    print("\nCalculating Emission Probabilities...")
    print("\nSample Emission Probabilities")
    for i, (key, value) in enumerate(emission_prob.items()):
        print(f"{key} -> {value:.6f}")
        if i >= 9:
            break

    return tagged_train, list(tag_counts.keys()), initial_probabilities, transition_prob, emission_prob


def get_emission_probability(tag, word, emission_prob):
    return emission_prob.get((tag, word.lower()), UNKNOWN_WORD_PROB)


def viterbi(words, tags, initial_probabilities, transition_prob, emission_prob):
    """
    Task 3: Viterbi decoder from scratch.
    Returns the Viterbi matrix and the most probable tag sequence.
    """
    viterbi_matrix = [{}]
    path = {}

    for tag in tags:
        init_prob = initial_probabilities.get(tag, 1e-10)
        emit_prob = get_emission_probability(tag, words[0], emission_prob)
        viterbi_matrix[0][tag] = init_prob * emit_prob
        path[tag] = [tag]

    for t in range(1, len(words)):
        viterbi_matrix.append({})
        new_path = {}

        for curr_tag in tags:
            emit_prob = get_emission_probability(curr_tag, words[t], emission_prob)
            best_prob = -1.0
            best_prev_tag = None

            for prev_tag in tags:
                trans_prob = transition_prob.get((prev_tag, curr_tag), 1e-10)
                probability = viterbi_matrix[t - 1][prev_tag] * trans_prob * emit_prob
                if probability > best_prob:
                    best_prob = probability
                    best_prev_tag = prev_tag

            viterbi_matrix[t][curr_tag] = best_prob
            new_path[curr_tag] = path[best_prev_tag] + [curr_tag]

        path = new_path

    best_final_tag = max(viterbi_matrix[-1], key=lambda tag: viterbi_matrix[-1][tag])
    return viterbi_matrix, path[best_final_tag]


def demonstrate_viterbi_implementation(test_sentences, tags, initial_probabilities, transition_prob, emission_prob):
    """Task 3: Confirm Viterbi works on a sample test sentence."""
    print("Viterbi decoder implemented from scratch.")
    print("\nAlgorithm steps:")
    print("  1. Initialize: score each tag for the first word using P(tag|<START>) x P(word|tag)")
    print("  2. Recurse: for each next word, keep the best path to every tag")
    print("  3. Terminate: return the tag sequence ending at the highest-scoring final tag")

    sample_words = None
    for words in test_sentences:
        if 5 <= len(words) <= 8:
            sample_words = words
            break
    if sample_words is None:
        sample_words = test_sentences[0][:7]

    gold_tags = [tag for _, tag in nltk.pos_tag(sample_words)]
    _, predicted_tags = viterbi(
        sample_words, tags, initial_probabilities, transition_prob, emission_prob
    )

    print("\nSample decode on a test sentence:")
    print(f"  Input : {' '.join(sample_words)}")
    print(f"  Output: {' '.join(predicted_tags)}")
    print(f"  Gold  : {' '.join(gold_tags)}")

    print("\nWord-level comparison:")
    for word, predicted, gold in zip(sample_words, predicted_tags, gold_tags):
        match = "OK" if predicted == gold else "DIFF"
        print(f"  {word:12s}  predicted={predicted:6s}  gold={gold:6s}  [{match}]")


def display_viterbi_results(sentence, viterbi_matrix, predicted_tags):
    """Task 4: Show predicted tags and a slice of the Viterbi matrix."""
    print("\n")
    print("=" * 70)
    print("TASK 4 : DOMAIN SPECIFIC SENTENCE")
    print("=" * 70)

    print("\nInput Sentence:")
    print(" ".join(sentence))
    print("\nRunning Viterbi Decoder...\n")

    print("=" * 70)
    print("FINAL PREDICTED TAG SEQUENCE")
    print("=" * 70)
    for word, tag in zip(sentence, predicted_tags):
        print(f"{word:15s} --> {tag}")

    print("\n")
    print("=" * 70)
    print("VITERBI MATRIX (SLICE)")
    print("=" * 70)

    matrix_df = pd.DataFrame(viterbi_matrix)
    matrix_df.index = sentence

    print(matrix_df.iloc[:, :10])
    print(f"\n[Showing 10 of {matrix_df.shape[1]} tag columns]")


def print_task5_explanation(num_words, num_tags):
    """Task 5: Explain dynamic programming and why Viterbi beats exhaustive search."""
    print("\n")
    print("=" * 70)
    print("TASK 5 : EXPLANATION")
    print("=" * 70)

    print("""
DYNAMIC PROGRAMMING NATURE OF VITERBI
-------------------------------------

The Viterbi algorithm uses dynamic programming. Instead of evaluating
every possible POS tag sequence, it stores the best probability of
reaching each tag at every word position in the Viterbi matrix.

Each cell depends only on:
1. The best previous state at position t - 1
2. The transition probability P(tag_t | tag_{t-1})
3. The emission probability P(word_t | tag_t)

Subproblems overlap: many tag sequences share the same prefix. Viterbi
reuses previously computed best-path scores rather than recomputing them,
which avoids redundant work and keeps runtime polynomial.
""")

    exhaustive = num_tags ** num_words
    viterbi_ops = num_words * (num_tags ** 2)

    print(f"""
WHY VITERBI IS PREFERRED OVER EXHAUSTIVE SEARCH
------------------------------------------------

For this example sentence:
  Sentence length (N) = {num_words}
  Number of POS tags (T) = {num_tags}

Exhaustive search must score every possible tag sequence:
  Complexity: T^N = {num_tags}^{num_words} = {exhaustive:,} sequences

Viterbi considers only the best partial path to each tag at each step:
  Complexity: O(N x T^2) = {num_words} x {num_tags}^2 = {viterbi_ops:,} operations

Viterbi is dramatically faster while still returning the globally most
probable tag sequence under the Markov assumptions of an HMM.
""")


def main():
    download_nltk_resources()

    train_sentences, test_sentences = load_and_split_dataset()
    _, tags, initial_probabilities, transition_prob, emission_prob = tag_and_build_probabilities(
        train_sentences
    )

    print("\n")
    print("=" * 70)
    print("TASK 3 : VITERBI IMPLEMENTATION")
    print("=" * 70)

    demonstrate_viterbi_implementation(
        test_sentences, tags, initial_probabilities, transition_prob, emission_prob
    )

    # Domain-specific movie review sentence (7 words)
    sentence = [
        "The",
        "movie",
        "was",
        "absolutely",
        "fantastic",
        "and",
        "engaging",
    ]

    viterbi_matrix, predicted_tags = viterbi(
        sentence, tags, initial_probabilities, transition_prob, emission_prob
    )

    display_viterbi_results(sentence, viterbi_matrix, predicted_tags)
    print_task5_explanation(len(sentence), len(tags))

    print("\n")
    print("=" * 70)
    print("ASSIGNMENT SUMMARY")
    print("=" * 70)
    print(f"Total Sentences Used        : {len(train_sentences) + len(test_sentences)}")
    print(f"Training Sentences          : {len(train_sentences)}")
    print(f"Testing Sentences           : {len(test_sentences)}")
    print(f"Unique POS Tags             : {len(tags)}")
    print(f"Transition Probabilities    : {len(transition_prob)}")
    print(f"Emission Probabilities      : {len(emission_prob)}")
    print("\nAssignment completed successfully.")


if __name__ == "__main__":
    main()
