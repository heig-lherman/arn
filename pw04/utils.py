import numpy as np
from matplotlib import pyplot as pl


def compute_fscore(matrix, c):
    selector = [x for x in range(matrix.shape[1]) if x != c]
    tp = matrix[c, c]
    fp = np.sum(matrix[selector, c])
    fn = np.sum(matrix[c, selector])
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def plot_confusion_matrix(conf_matrix, classes_names):
    fig, ax = pl.subplots()
    ax.imshow(conf_matrix, cmap="viridis")

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(classes_names)), labels=classes_names)
    ax.set_yticks(np.arange(len(classes_names)), labels=classes_names)

    # Loop over data dimensions and create text annotations.
    for i in range(len(classes_names)):
        for j in range(len(classes_names)):
            ax.text(j, i, conf_matrix[i, j], fontweight="normal", fontsize="small", ha="center", va="center", c="r")

    fig.tight_layout()
    pl.show()

    for i in range(len(classes_names)):
        print(f"F-score {i}: {compute_fscore(conf_matrix, i)}")
