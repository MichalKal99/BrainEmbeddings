import os
import numpy as np

def order_trajectories_chronologically(trajectories_and_ts, ordered_markers):
    # get a unique list with the words in the right order
    ordered_words = list(dict.fromkeys(ordered_markers))

    # organize the words in the right order
    words_with_traject_start_and_end_points_ordered = []
    for word in ordered_words:
        for item in trajectories_and_ts:
            if item.get("word") == word:
                entry = item
                break

        words_with_traject_start_and_end_points_ordered.append(entry)

    # just to be sure, if this block prints anything, then we have a problem
    for i, word in enumerate(words_with_traject_start_and_end_points_ordered):
        if word.get("word") != ordered_words[i]:
            print("BATILDOOOOOOOOO")
            print(word.get("word"), ordered_words[i])
            raise ValueError("Empty data for one of the words")

    return words_with_traject_start_and_end_points_ordered


def read_trajectories(folder, format="_predicted.matrix"):
    if format not in ["_predicted.matrix", "_predicted.txt"]:
        print("Format not recognized: {}".format(format))
        return None

    word_list_trajectories = []
    for file in os.listdir(folder):
        if file.endswith("_predicted.txt") and format == "_predicted.txt":
            with open(os.path.join(folder, file), "r") as f:
                vocal_fold_params = None
                vocal_tract_params = None
                trajectories = None
                for i, line in enumerate(f):
                    if i > 7:
                        # line containing vocal_fold_params
                        if not i % 2:
                            vocal_fold_params = [float(num) for num in line.split()]
                        else:
                            vocal_tract_params = [float(num) for num in line.split()]
                            trajectory = np.append(
                                vocal_fold_params, vocal_tract_params
                            )
                            if trajectories is None:
                                trajectories = trajectory
                            else:
                                trajectories = np.vstack((trajectories, trajectory))

                filename = file.split("_")[1]
                entry = {"name": filename, "trajectories": trajectories}
                word_list_trajectories.append(entry)

        if file.endswith("_predicted.matrix") and format == "_predicted.matrix":
            with open(os.path.join(folder, file), "r") as f:
                trajectories = []
                for i, line in enumerate(f):
                    trajectory = [float(num) for num in line.split()]
                    trajectories.append(trajectory)

                filename = file.split("_")[1]
                entry = {"name": filename, "trajectories": trajectories}
                word_list_trajectories.append(entry)

    return word_list_trajectories
