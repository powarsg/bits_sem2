"""
===========================================================
BITS WILP - Deep Reinforcement Learning
Lab Assignment 1 - Part 1 (MAB)

Group Number : 116

Author:
===========================================================

This program implements:

Task 1:
    - Synthetic dataset generation

Task 2:
    - Immediate Exploitation Strategy

Task 3:
    - Epsilon Greedy Strategy
      epsilon = 10%
      epsilon = 1%
      epsilon = 50%

Task 4:
    - UCB1 Algorithm

Task 5:
    - Comparative Analysis
      Cumulative Reward Graph

===========================================================
"""

# =========================================================
# Import Required Libraries
# =========================================================

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# Assignment Parameters
# =========================================================

GROUP_NO = 118

# Reproducibility
random.seed(GROUP_NO)
np.random.seed(GROUP_NO)

# =========================================================
# Task 1 : Dataset Creation
# =========================================================

# Number of medicines
K = (GROUP_NO % 3) + 5

print("=" * 60)
print("GROUP NUMBER :", GROUP_NO)
print("TOTAL MEDICINES :", K)
print("=" * 60)

# ---------------------------------------------------------
# Hidden success probability for each medicine
#
# Formula:
#
# Pi = 0.4 + ((G+i) mod 6)*0.07
# ---------------------------------------------------------

hidden_probabilities = []

for i in range(K):

    p = 0.4 + (((GROUP_NO + i) % 6) * 0.07)

    hidden_probabilities.append(round(p, 2))

print("\nHidden Success Probabilities")
print("--------------------------------")

for i, p in enumerate(hidden_probabilities):
    print(f"Medicine {i} : {p}")

# ---------------------------------------------------------
# Generate Base Dataset
#
# severity = (patient_id % 5)+1
#
# Total patients = 1000
# ---------------------------------------------------------

TOTAL_PATIENTS = 1000

dataset = pd.DataFrame()

dataset["patient_id"] = range(TOTAL_PATIENTS)

dataset["severity_score"] = \
    dataset["patient_id"].apply(
        lambda x: (x % 5) + 1
    )

print("\nFirst 10 Dataset Rows")
print(dataset.head(10))

# =========================================================
# Environment Simulation Function
# =========================================================

def simulate_patient(medicine_id, severity):
    """
    Simulates treatment outcome for a patient.

    Parameters
    ----------
    medicine_id : int
        Selected medicine

    severity : int
        Disease severity (1-5)

    Returns
    -------
    outcome : int
        1 = recovered
        0 = not recovered

    reward : float
        Utility score
    """

    # Hidden probability of medicine
    success_probability = hidden_probabilities[medicine_id]

    # Recovery simulation
    outcome = np.random.binomial(
        n=1,
        p=success_probability
    )

    # Utility Score
    reward = outcome * (1 - severity / 10)

    return outcome, reward


# =========================================================
# Task 2
# Immediate Exploitation Strategy
# =========================================================

def immediate_exploitation():

    """
    Strategy:

    Step 1:
        Test each medicine exactly 10 times

    Step 2:
        Calculate success rate

    Step 3:
        Select best medicine

    Step 4:
        Use only that medicine for
        remaining patients
    """

    rewards_history = []

    medicine_successes = np.zeros(K)
    medicine_counts = np.zeros(K)

    cumulative_reward = 0

    patient_counter = 0

    # ----------------------------------
    # Initial exploration
    # ----------------------------------

    for medicine in range(K):

      #print("Medicine selected : ", medicine)

        for _ in range(10):

            severity = (
                patient_counter % 5
            ) + 1

            outcome, reward = simulate_patient(
                medicine,
                severity
            )

            medicine_successes[medicine] += outcome
            medicine_counts[medicine] += 1

            cumulative_reward += reward

            rewards_history.append(
                cumulative_reward
            )
          
            patient_counter += 1

    # ----------------------------------
    # Identify best medicine
    # ----------------------------------

    # Calculate success rates for each medicine
    success_rates = (
        medicine_successes /
        medicine_counts
    )

    # Print success rates for all medicines
    print("\nSuccess Rates for Each Medicine:")
    for i, rate in enumerate(success_rates):
        print(f"Medicine {i}: {rate:.2f}")

    # Select the medicine with the highest success rate
    best_medicine = np.argmax(
        success_rates
    )

    # Print the best medicine selected
    print("\nBest Medicine Selected:")
    print(f"Medicine {best_medicine} with success rate {success_rates[best_medicine]:.2f}")

    # ----------------------------------
    # Exploit best medicine
    # ----------------------------------

    while patient_counter < TOTAL_PATIENTS:

        severity = (
            patient_counter % 5
        ) + 1

        outcome, reward = simulate_patient(
            best_medicine,
            severity
        )

        cumulative_reward += reward

        rewards_history.append(
            cumulative_reward
        )

        patient_counter += 1

    return rewards_history


# =========================================================
# Task 3
# Epsilon Greedy
# =========================================================

def epsilon_greedy(epsilon):
    """
    Epsilon Greedy Strategy

    With probability epsilon:
        Explore

    With probability (1-epsilon):
        Exploit

    Parameters
    ----------
    epsilon : float

    Returns
    -------
    rewards_history
    """

    rewards_history = []

    successes = np.zeros(K)
    counts = np.zeros(K)

    cumulative_reward = 0

    for patient in range(TOTAL_PATIENTS):

        severity = (patient % 5) + 1

        # ----------------------------------
        # Ensure every medicine
        # gets at least one trial
        # ----------------------------------

        if patient < K:

            medicine = patient

        else:

            # Exploration
            if random.random() < epsilon:

                medicine = random.randint(
                    0,
                    K - 1
                )

            # Exploitation
            else:

                averages = np.divide(
                    successes,
                    counts,
                    out=np.zeros_like(
                        successes
                    ),
                    where=counts != 0
                )

                medicine = np.argmax(
                    averages
                )

        outcome, reward = simulate_patient(
            medicine,
            severity
        )

        successes[medicine] += outcome
        counts[medicine] += 1

        cumulative_reward += reward

        rewards_history.append(
            cumulative_reward
        )

    return rewards_history


# =========================================================
# Task 4
# UCB1 Algorithm
# =========================================================

def ucb1():
    """
    Upper Confidence Bound

    Formula:

    UCB =
    average_reward +
    sqrt(
        (2*ln(total_trials))
        /
        arm_trials
    )
    """

    rewards_history = []

    successes = np.zeros(K)
    counts = np.zeros(K)

    cumulative_reward = 0

    # ----------------------------------
    # Give each medicine
    # one initial trial
    # ----------------------------------

    for medicine in range(K):

        severity = (medicine % 5) + 1

        outcome, reward = simulate_patient(
            medicine,
            severity
        )

        successes[medicine] += outcome
        counts[medicine] += 1

        cumulative_reward += reward

        rewards_history.append(
            cumulative_reward
        )

    # ----------------------------------
    # UCB iterations
    # ----------------------------------

    for patient in range(K, TOTAL_PATIENTS):

        severity = (
            patient % 5
        ) + 1

        ucb_values = []

        for arm in range(K):

            average_reward = (
                successes[arm]
                /
                counts[arm]
            )

            confidence = np.sqrt(
                (
                    2 *
                    np.log(patient + 1)
                )
                /
                counts[arm]
            )

            ucb = (
                average_reward
                +
                confidence
            )

            ucb_values.append(ucb)

        medicine = np.argmax(
            ucb_values
        )

        outcome, reward = simulate_patient(
            medicine,
            severity
        )

        successes[medicine] += outcome
        counts[medicine] += 1

        cumulative_reward += reward

        rewards_history.append(
            cumulative_reward
        )

    return rewards_history


# =========================================================
# Execute All Strategies
# =========================================================

print("\nRunning Immediate Exploitation...")
immediate_rewards = immediate_exploitation()

print("\nRunning Epsilon Greedy (10%)...")
eps10_rewards = epsilon_greedy(epsilon=0.10)

print("\nRunning Epsilon Greedy (1%)...")
eps1_rewards = epsilon_greedy(epsilon=0.01)

print("\nRunning Epsilon Greedy (50%)...")
eps50_rewards = epsilon_greedy(epsilon=0.50)

print("\nRunning UCB1...")
ucb_rewards = ucb1()

# =========================================================
# Final Rewards
# =========================================================

print("\n")
print("=" * 60)
print("FINAL CUMULATIVE REWARDS")
print("=" * 60)

print(
    f"Immediate Exploitation : "
    f"{immediate_rewards[-1]:.2f}"
)

print(
    f"Epsilon Greedy 10% : "
    f"{eps10_rewards[-1]:.2f}"
)

print(
    f"Epsilon Greedy 1% : "
    f"{eps1_rewards[-1]:.2f}"
)

print(
    f"Epsilon Greedy 50% : "
    f"{eps50_rewards[-1]:.2f}"
)

print(
    f"UCB1 : "
    f"{ucb_rewards[-1]:.2f}"
)

# =========================================================
# Task 5
# Comparative Graph
# =========================================================

plt.figure(figsize=(12, 6))

plt.plot(
    immediate_rewards,
    label="Immediate Exploitation"
)

plt.plot(
    eps10_rewards,
    label="Epsilon Greedy 10%"
)

plt.plot(
    eps1_rewards,
    label="Epsilon Greedy 1%"
)

plt.plot(
    eps50_rewards,
    label="Epsilon Greedy 50%"
)

plt.plot(
    ucb_rewards,
    label="UCB1"
)

plt.xlabel(
    "Number of Patients"
)

plt.ylabel(
    "Cumulative Reward"
)

plt.title(
    "Cumulative Reward Comparison"
)

plt.legend()

plt.grid(True)

plt.show()

# =========================================================
# Best Performing Strategy
# =========================================================

results = {
    "Immediate":
        immediate_rewards[-1],

    "Epsilon-10%":
        eps10_rewards[-1],

    "Epsilon-1%":
        eps1_rewards[-1],

    "Epsilon-50%":
        eps50_rewards[-1],

    "UCB1":
        ucb_rewards[-1]
}

best_strategy = max(
    results,
    key=results.get
)

print("\nBest Strategy:")
print(best_strategy)

print(
    "\nDetailed Results:"
)

for strategy, value in results.items():

    print(
        f"{strategy:20s}"
        f"{value:.2f}"
    )
