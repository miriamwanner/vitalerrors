import matplotlib.pyplot as plt
import numpy as np

def plot_v1():

    # X-axis range: total claims from 0 to 30
    x = np.arange(0, 31)

    # Precision: (x - 1) / n, where n is 30
    # n = 30
    precision = ((x - 1) / x) * 100
    # precision = np.clip(precision, 0, 1) * 100  # scale to percentage

    # Recall @ 10: step increases until 90, then stays constant
    recall = np.zeros_like(x, dtype=float)
    recall_steps = {
        2: 10,
        3: 20,
        4: 30,
        5: 40,
        6: 50,
        7: 60,
        8: 70,
        9: 80,
        10: 90
    }
    for i, val in enumerate(x):
        if val in recall_steps:
            recall[i] = recall_steps[val]
        elif val > 10:
            recall[i] = 90
        else:
            recall[i] = recall[i-1] if i > 0 else 0

    # Plotting
    plt.figure(figsize=(6, 4))
    plt.plot(x, recall, label="Recall @ 10", marker='o')
    plt.plot(x, precision, label="Precision", marker='o')

    plt.xlim(0, 30)
    plt.ylim(0, 100)
    plt.xlabel("Total Claims")
    plt.ylabel("Expected Score")
    plt.title("One key subclaim wrong")
    plt.legend()
    # plt.grid(True)
    plt.tight_layout()

    plt.show()


def precision():

    # X-axis range: total claims from 0 to 30
    x = np.arange(0, 31)

    # Precision: (x - 1) / n, where n is 30
    # n = 30
    precision1 = ((x - 1) / x) * 100
    # precision = np.clip(precision, 0, 1) * 100  # scale to percentage

    precision2 = ((x - 2) / x) * 100
    # precision = np.clip(precision, 0, 1) * 100  # scale to percentage

    precision3 = ((x - 3) / x) * 100
    # precision = np.clip(precision, 0, 1) * 100  # scale to percentage

    # Plotting
    plt.figure(figsize=(6, 4))
    plt.plot(x, precision1, label="1 incorrect", marker='o')
    plt.plot(x, precision2, label="2 incorrect", marker='o')
    plt.plot(x, precision3, label="3 incorrect", marker='o')

    plt.xlim(0, 30)
    plt.ylim(0, 100)
    plt.xlabel("Total Claims")
    plt.ylabel("Expected Factscore")
    plt.title("Precision")
    plt.legend()
    # plt.grid(True)
    plt.tight_layout()

    plt.show()

def recall():

    # X-axis range: total claims from 0 to 30
    x = np.arange(0, 31)

    # Precision: (x - 1) / n, where n is 30
    # n = 30
    # recall1 = ((x - 1) / 10) * 100
    recall1 = np.where(x < 10, ((x - 1) / 10) * 100, 90)
    # precision = np.clip(precision, 0, 1) * 100  # scale to percentage

    recall2 = np.where(x < 10, ((x - 2) / 10) * 100, 80)
    # precision = np.clip(precision, 0, 1) * 100  # scale to percentage

    recall3 = np.where(x < 10, ((x - 3) / 10) * 100, 70)
    # precision = np.clip(precision, 0, 1) * 100  # scale to percentage

    # Plotting
    plt.figure(figsize=(6, 4))
    plt.plot(x, recall1, label="1 missing", marker='o')
    plt.plot(x, recall2, label="2 missing", marker='o')
    plt.plot(x, recall3, label="3 missing", marker='o')

    plt.xlim(0, 30)
    plt.ylim(0, 100)
    plt.xlabel("Total Claims")
    plt.ylabel("Expected Nuggets Recall")
    plt.title("Recall with 10 Nuggets")
    plt.legend()
    # plt.grid(True)
    plt.tight_layout()

    plt.show()


def main():
    precision()
    recall()


if __name__ =="__main__":
    main()