import csv
import numpy as np
import os
import itertools
import tensorflow as tf

LABEL_NAMES = ["4h", "BL", "POD1", "POD2_"]
EXCLUDED_VIDEOS = [
    "SEG_3418",     # VI-15
    "SEG_3434",     # GFAP-31
    "SEG_3426",     # VI-55
    "SEG_3398",     # VI-15
    "SEG_3400",     # VI-21
    "SEG_3411",     # GFAP-23
    "SEG_3439",     # GFAP-15
    "SEG_3441",     # GFAP-21
    "SEG_3446",     # GFAP-32
    "SEG_3459",     # VI-15
    "SEG_3467",     # VI-55
    "SEG_3476",     # GFAP-32
]


def read_csv_to_ndarray(csv_path: str) -> np.array:
    """
    read single .csv file to np.array
    """
    with open(csv_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file)

        # could be refactored
        csv_lines = []
        for line in csv_reader:
            csv_lines.append(line)

    # omit frame number
    for i, line in enumerate(csv_lines):
        csv_lines[i] = line[1:]

    # return everything but header lines
    return np.array(csv_lines[3:], dtype=np.float32).T


def calculate_relative_positions(pos_array: np.ndarray) -> np.array:
    """
    Calculate relative changes on absolute positions.
    """
    rel_array = pos_array.copy()

    for i, row in enumerate(rel_array):
        # skip every third row (contains confidence not coordinates)
        if (i + 1) % 3 == 0:
            continue

        # calculate relative change between frames
        prev_elem = row[0]
        for j, elem in enumerate(row):
            rel_array[i, j] = elem - prev_elem
            prev_elem = elem

    return rel_array


def get_joints(csv_path: str) -> list:
    with open(csv_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file)

        # could be refactored
        csv_lines = []
        for line in csv_reader:
            csv_lines.append(line)

    # get all joint names
    joints_with_duplicates = csv_lines[1]

    # remove duplicates
    joints = []
    for joint in joints_with_duplicates:
        if not joint in joints:
            joints.append(joint)

    return joints


def create_label_lookup(
    path: str,
) -> dict:
    """
    Create lookup with label for each video
    """
    label_lookup = {}

    for label in LABEL_NAMES:
        for videoname in os.listdir(os.path.join(path, label)):
            label_lookup[videoname[:-4]] = LABEL_NAMES.index(label)

    return label_lookup


def load_data_and_labels(
    datapath: str,
    load_only_excluded: bool = False,
) -> tuple[list, list, list]:
    """
    Load .csv files from DLC prediction.
    Uses the filtered data.
    """
    X_pos = []
    X_rel = []
    y = []

    label_lookup = create_label_lookup()

    for filename in os.listdir(datapath):
        if load_only_excluded:
            load_video = filename[:8] in EXCLUDED_VIDEOS and filename.endswith(
                "filtered.csv")
        else:
            load_video = not filename[:8] in EXCLUDED_VIDEOS and filename.endswith(
                "filtered.csv")

        if load_video:
            pos_array = read_csv_to_ndarray(os.path.join(datapath, filename))
            rel_array = calculate_relative_positions(pos_array)
            current_shape = pos_array.shape
            pos_array = pos_array.reshape(
                (int(current_shape[0] / 3), 3, current_shape[1]))
            rel_array = rel_array.reshape(
                (int(current_shape[0] / 3), 3, current_shape[1]))
            pos_array = np.swapaxes(pos_array, 0, 1)
            rel_array = np.swapaxes(rel_array, 0, 1)
            X_pos.append(pos_array)
            X_rel.append(rel_array)
            y.append(label_lookup[filename[:8]])

    return X_pos, X_rel, y


def window_data(X_pos: list, X_rel: list, y: list, num_frames: int = 250, start_buffer: int = 0, end_buffer: int = 0, overlap: int = 150) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Slice data into smaller windows of size num_frames with overlapping frames of overlap
    """
    X_pos_split = []
    X_rel_split = []
    y_split = []

    for index in range(len(y)):
        for start, end in zip(range(start_buffer, len(X_pos[index][0, 0]) - end_buffer - num_frames, num_frames - overlap), range(start_buffer + num_frames, len(X_pos[index][0, 0]) - end_buffer, num_frames - overlap)):
            X_pos_split.append(X_pos[index][:, :, start:end])
            X_rel_split.append(X_rel[index][:, :, start:end])
            y_split.append(y[index])

    return np.array(X_pos_split), np.array(X_rel_split), np.array(y_split)


def np_train_val_test_split(array: np.array, train_pct: float, val_pct: float, ds_length: int) -> tuple[np.array, np.array, np.array]:
    """
    Split data into train, val and test sets. array has to be shuffeled before calling this method.
    """
    ds_length = float(ds_length)

    train_size = int(train_pct * ds_length)
    val_size = int(val_pct * ds_length)

    array_train = array[0:train_size]
    array_val = array[train_size:train_size + val_size]
    array_test = array[train_size + val_size:]

    return array_train, array_val, array_test
