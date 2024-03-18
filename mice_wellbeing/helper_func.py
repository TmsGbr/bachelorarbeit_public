import tensorflow as tf
import keras
import os
import numpy as np

from mice_wellbeing import data_handler


def predict_videos(model: keras.Model, videos: list = None, input_mode: str = "comb", batch_size: int = 4, num_frames=250, overlap=150, start_buffer=50, end_buffer=50, datapath: str = "/home/thomas/bachelorarbeit/Mice_Wellbeing-Thomas_Gebauer-2023-11-18/videos") -> dict:
    """
    Use this to predict excluded videos
    """
    if videos is None:
        videos = data_handler.EXCLUDED_VIDEOS

    label_lookup = data_handler.create_label_lookup()

    result_dict = {}

    X_pos_to_eval = []
    X_rel_to_eval = []
    y_to_eval = []

    for filename in os.listdir(datapath):
        if filename[:8] in videos and filename.endswith("filtered.csv"):
            pos_array = data_handler.read_csv_to_ndarray(
                os.path.join(datapath, filename))
            rel_array = data_handler.calculate_relative_positions(pos_array)
            current_shape = pos_array.shape
            pos_array = pos_array.reshape(
                (int(current_shape[0] / 3), 3, current_shape[1]))
            rel_array = rel_array.reshape(
                (int(current_shape[0] / 3), 3, current_shape[1]))
            pos_array = np.swapaxes(pos_array, 0, 1)
            rel_array = np.swapaxes(rel_array, 0, 1)
            y = label_lookup[filename[:8]]

            X_pos_split, X_rel_split, y_split = data_handler.window_data([pos_array], [
                                                                         rel_array], [y], num_frames=num_frames, overlap=overlap, start_buffer=start_buffer, end_buffer=end_buffer)

            for row in list(X_pos_split):
                X_pos_to_eval.append(row)
                y_to_eval.append(y)

            for row in list(X_rel_split):
                X_rel_to_eval.append(row)

            if input_mode == "pos":
                y_pred = model.predict(X_pos_split, batch_size=batch_size)
            elif input_mode == "rel":
                y_pred = model.predict(X_rel_split, batch_size=batch_size)
            elif input_mode == "comb":
                X_comb_split = tf.data.Dataset.from_tensor_slices(
                    ((X_pos_split, X_rel_split), y_split), name="comb_eval").batch(batch_size)
                y_pred = model.predict(X_comb_split)

            result_dict[filename[:8]] = {
                "y": y,
                "y_pred": y_pred,
                "y_count": np.unique(y_pred.argmax(axis=1), return_counts=True),
            }
            #print(
            #    f"Video: {filename[:8]}\nKlasse: {y}\nPredictions: {result_dict[filename[:8]]['y_count']}")

    X_pos_to_eval = np.array(X_pos_to_eval)
    X_rel_to_eval = np.array(X_rel_to_eval)
    y_to_eval = np.array(y_to_eval)

    if input_mode == "pos":
        eval_ds = tf.data.Dataset.from_tensor_slices(
            (X_pos_to_eval, y_to_eval))
    elif input_mode == "rel":
        eval_ds = tf.data.Dataset.from_tensor_slices(
            (X_rel_to_eval, y_to_eval))
    elif input_mode == "comb":
        eval_ds = tf.data.Dataset.from_tensor_slices(
            ((X_pos_to_eval, X_rel_to_eval), y_to_eval))

    eval_ds = eval_ds.batch(batch_size)
    print(model.evaluate(eval_ds))

    return result_dict
