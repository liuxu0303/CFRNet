import numpy as np
import argparse
import glob
from os.path import join
import tqdm


def FLAGS():
    parser = argparse.ArgumentParser("""Event Depth Data Evaluation.""")

    parser.add_argument("--target_dataset",
                        default="./save_checkpoint/CFRNet_DSEC_epoch20_bs_3_lr_00032/test_out/gt")
    parser.add_argument("--predictions_dataset",
                        default="./save_checkpoint/CFRNet_DSEC_epoch20_bs_3_lr_00032/test_out/pred_depth")
    parser.add_argument("--clip_distance", type=float, default=50.0)

    flags = parser.parse_args()

    return flags


metrics_keywords = [
    f"_abs_rel_diff",
    f"_squ_rel_diff",
    f"_RMS_linear",
    f"_RMS_log",
    f"_SILog",
    f"_mean_target_depth",
    f"_median_target_depth",
    f"_mean_prediction_depth",
    f"_median_prediction_depth",
    f"_mean_depth_error",
    f"_median_diff",
    f"_threshold_delta_1.25",
    f"_threshold_delta_1.25^2",
    f"_threshold_delta_1.25^3",
    f"_10_mean_target_depth",
    f"_10_median_target_depth",
    f"_10_mean_prediction_depth",
    f"_10_median_prediction_depth",
    f"_10_abs_rel_diff",
    f"_10_squ_rel_diff",
    f"_10_RMS_linear",
    f"_10_RMS_log",
    f"_10_SILog",
    f"_10_mean_depth_error",
    f"_10_median_diff",
    f"_10_threshold_delta_1.25",
    f"_10_threshold_delta_1.25^2",
    f"_10_threshold_delta_1.25^3",
    f"_20_abs_rel_diff",
    f"_20_squ_rel_diff",
    f"_20_RMS_linear",
    f"_20_RMS_log",
    f"_20_SILog",
    f"_20_mean_target_depth",
    f"_20_median_target_depth",
    f"_20_mean_prediction_depth",
    f"_20_median_prediction_depth",
    f"_20_mean_depth_error",
    f"_20_median_diff",
    f"_20_threshold_delta_1.25",
    f"_20_threshold_delta_1.25^2",
    f"_20_threshold_delta_1.25^3",
    f"_30_abs_rel_diff",
    f"_30_squ_rel_diff",
    f"_30_RMS_linear",
    f"_30_RMS_log",
    f"_30_SILog",
    f"_30_mean_target_depth",
    f"_30_median_target_depth",
    f"_30_mean_prediction_depth",
    f"_30_median_prediction_depth",
    f"_30_mean_depth_error",
    f"_30_median_diff",
    f"_30_threshold_delta_1.25",
    f"_30_threshold_delta_1.25^2",
    f"_30_threshold_delta_1.25^3",
]


def prepare_depth_data(target, prediction, clip_distance):
    min_depth = 4.59
    max_depth = clip_distance

    prediction[np.isinf(prediction)] = max_depth
    prediction[np.isnan(prediction)] = min_depth

    depth_mask = (np.ones_like(target) > 0)
    valid_mask = np.logical_and(target > min_depth, target < max_depth)
    valid_mask = np.logical_and(depth_mask, valid_mask)

    return target, prediction, valid_mask



def add_to_metrics(metrics, target_, prediction_, mask, prefix=""):
    if len(metrics) == 0:
        metrics = {k: 0 for k in metrics_keywords}

    eps = 1e-5

    target = target_[
        mask]  # np.where(mask, target_, np.max(target_[~np.isnan(target_)]))# target_[mask] but without lossing shape
    prediction = prediction_[
        mask]  # np.where(mask, prediction_, np.max(target_[~np.isnan(target_)]))# prediction_[mask] but without lossing shape

    # thresholds
    # ratio = np.max(np.stack([target/(prediction+eps),prediction/(target+eps)]), axis=0)
    ratio = np.maximum((target / (prediction + eps)), (prediction / (target + eps)))

    new_metrics = {}
    new_metrics[f"{prefix}threshold_delta_1.25"] = np.mean(ratio <= 1.25)
    new_metrics[f"{prefix}threshold_delta_1.25^2"] = np.mean(ratio <= 1.25 ** 2)
    new_metrics[f"{prefix}threshold_delta_1.25^3"] = np.mean(ratio <= 1.25 ** 3)

    # abs diff
    # log_diff = np.log(target+eps)-np.log(prediction+eps)
    log_diff = np.log(prediction + eps) - np.log(target + eps)
    # log_diff = np.abs(log_target - log_prediction)
    abs_diff = np.abs(target - prediction)

    new_metrics[f"{prefix}abs_rel_diff"] = (abs_diff / (target + eps)).mean()
    new_metrics[f"{prefix}squ_rel_diff"] = (abs_diff ** 2 / (target ** 2 + eps)).mean()
    new_metrics[f"{prefix}RMS_linear"] = np.sqrt((abs_diff ** 2).mean())
    new_metrics[f"{prefix}RMS_log"] = np.sqrt((log_diff ** 2).mean())
    new_metrics[f"{prefix}SILog"] = (log_diff ** 2).mean() - (log_diff.mean()) ** 2
    new_metrics[f"{prefix}mean_target_depth"] = target.mean()
    new_metrics[f"{prefix}median_target_depth"] = np.median(target)
    new_metrics[f"{prefix}mean_prediction_depth"] = prediction.mean()
    new_metrics[f"{prefix}median_prediction_depth"] = np.median(prediction)
    new_metrics[f"{prefix}mean_depth_error"] = abs_diff.mean()
    new_metrics[f"{prefix}median_diff"] = np.abs(np.median(target) - np.median(prediction))

    for k, v in new_metrics.items():
        metrics[k] += v

    return metrics


if __name__ == "__main__":
    flags = FLAGS()

    # predicted labels
    prediction_files = sorted(glob.glob(join(flags.predictions_dataset, '*.npy')))
    prediction_files = prediction_files[flags.prediction_offset:]

    target_files = sorted(glob.glob(join(flags.target_dataset, '*.npy')))
    target_files = target_files[flags.target_offset:]

    # Information about the dataset length
    print("len of prediction files", len(prediction_files))
    print("len of target files", len(target_files))

    assert len(prediction_files) > 0
    assert len(target_files) > 0

    metrics = {}

    num_it = len(target_files)
    for idx in tqdm.tqdm(range(num_it)):
        p_file, t_file = prediction_files[idx], target_files[idx]

        # Read absolute scale ground truth
        target_depth = np.load(t_file)
        target_depth = np.squeeze(target_depth)
        # print(target_depth.shape)

        # Read predicted depth data
        predicted_depth = np.load(p_file)
        predicted_depth = np.squeeze(predicted_depth)

        # Convert to the correct scale
        target_depth, predicted_depth, valid_mask = prepare_depth_data(target_depth, predicted_depth,
                                                                       flags.clip_distance)

        assert predicted_depth.shape == target_depth.shape

        metrics = add_to_metrics(metrics, target_depth, predicted_depth, valid_mask, prefix="_")

        for depth_threshold in [10, 20, 30]:
            depth_threshold_mask = (np.nan_to_num(target_depth) < depth_threshold)
            add_to_metrics(-1, metrics, target_depth, predicted_depth, valid_mask & depth_threshold_mask,
                           prefix=f"_{depth_threshold}_")

    # pprint({k: v / num_it for k, v in metrics.items()})
    {print("%s : %f" % (k, v / num_it)) for k, v in metrics.items()}