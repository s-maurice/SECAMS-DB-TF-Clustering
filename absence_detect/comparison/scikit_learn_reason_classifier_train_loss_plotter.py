import matplotlib.pyplot as plt

file = open("scikit_learn_reason_classifier_train_loss.txt", "r")
file = file.readlines()

iterations = []
[iterations.append(line) for line in file if line[0] == "I"]

iterations_split = []
cur_iter_seq = None
for line in iterations:

    cur_iter = line.split(",")
    cur_iter[0] = int(cur_iter[0].split(" ")[1])
    cur_iter[1] = cur_iter[1].split(" ")[3][:-1]

    # Replace inf with high number, or convert to float
    if cur_iter[1] == "inf":
        cur_iter[1] = 100
    else:
        cur_iter[1] = float(cur_iter[1])

    if cur_iter[0] == 1:
        if cur_iter_seq is not None:
            iterations_split.append(cur_iter_seq)
        cur_iter_seq = []

    cur_iter_seq.append(cur_iter)
iterations_split.append(cur_iter_seq)

for iter in iterations_split[2]:
    print(iter)

fig, ax = plt.subplots(nrows=1, ncols=len(iterations_split), sharey=True)
for subplot, iter_data, idx in zip(ax, iterations_split, range(len(ax))):
    iter_data = list(map(list, zip(*iter_data)))  # Transpose
    subplot.plot(iter_data[0], iter_data[1])
    subplot.set_ylim((0, 3))
    subplot.set_xlabel("Steps")
    subplot.grid(axis="y")
    subplot.set_title("Cross Validation Fold " + str(idx))
ax[0].set_ylabel("Log Loss")
fig.canvas.set_window_title("Deviation Comparison")
fig.suptitle("SciKit Learn Reason Classifier Training Loss")
plt.show()

