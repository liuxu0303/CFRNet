import os
import json
import logging
import argparse
import torch
from model.model import *
from model.loss import *
from torch.utils.data import DataLoader, ConcatDataset
from utils.data import concatenate_datasets
from data_loader.dataset_dsec import Sequence
from data_loader.data_loaders import HDF5DataLoader
from os.path import join
import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np


os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

logging.basicConfig(level=logging.INFO, format='')


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_colormap(img, color_mapper):
    # color_map = np.nan_to_num(img[0])
    # print("max min color map: ", np.max(color_map), np.min(color_map))
    img = np.nan_to_num(img, nan=1)
    color_map_inv = np.ones_like(img[0]) * np.amax(img[0]) - img[0]
    color_map_inv = color_map_inv / np.amax(color_map_inv)
    color_map_inv = np.nan_to_num(color_map_inv)
    color_map_inv = color_mapper.to_rgba(color_map_inv)
    color_map_inv[:, :, 0:3] = color_map_inv[:, :, 0:3][..., ::-1]
    return color_map_inv


# def disp_to_depth(disp, min_depth=2.0, max_depth=80.0):
def disp_to_depth(disp, min_depth=4.59, max_depth=50.0):
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return depth


def main(config, initial_checkpoint, output_folder):

    L = 1

    dataset_type, base_folder, root_folder = {}, {}, {}

    step_size = {}
    delta_t_ms = {}
    clip_distance = {}
    num_bins = {}

    if output_folder:
        ensure_dir(output_folder)
        pred_depth_dir = join(output_folder, "pred_depth")
        gt_dir = join(output_folder, "gt")
        inputs_dir = join(output_folder, "inputs")
        color_map_dir = join(output_folder, "color_map")
        ensure_dir(pred_depth_dir)
        ensure_dir(gt_dir)
        ensure_dir(inputs_dir)
        ensure_dir(color_map_dir)



    for split in ['train', 'validation']:
        dataset_type[split] = config['data_loader'][split]['type']
        base_folder[split] = join(config['data_loader'][split]['base_folder'])
        root_folder[split] = join(config['data_loader'][split]['root_folder'])

        try:
            delta_t_ms[split] = config['data_loader'][split]['delta_t_ms']
        except KeyError:
            delta_t_ms[split] = 50

        try:
            step_size[split] = config['data_loader'][split]['step_size']
        except KeyError:
            step_size[split] = 1

        try:
            clip_distance[split] = config['data_loader'][split]['clip_distance']
        except KeyError:
            clip_distance[split] = 100.0

        try:
            num_bins[split] = config['data_loader'][split]['num_bins']
        except KeyError:
            num_bins[split] = 5

    normalize = config['data_loader'].get('normalize', True)

    sequence_kwargs = {'root_folder': root_folder['validation'],
                       'mode': 'validation',
                       'delta_t_ms': delta_t_ms['validation'],
                       'sequence_length': L,
                       'transform': None,
                       'step_size': L,
                       'clip_distance': clip_distance['validation'],
                       'num_bins': num_bins['validation'],
                       'normalize': normalize}

    test_dataset = concatenate_datasets(base_folder['validation'], Sequence, sequence_kwargs,
                                        dataset_idx_flag=True)


    model = eval(config['arch'])(config['model'])

    if initial_checkpoint is not None:
        print('Loading initial model weights from: {}'.format(initial_checkpoint))
        checkpoint = torch.load(initial_checkpoint)
        model.load_state_dict(checkpoint['state_dict'])

    gpu = torch.device('cuda:' + str(config['gpu']))
    model.to(gpu)

    model.eval()

    N = len(test_dataset)

    print('N: ' + str(N))

    # construct color mapper
    normalizer = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    color_mapper_overall = cm.ScalarMappable(norm=normalizer, cmap='magma')

    with torch.no_grad():
        idx = 0
        prev_dataset_idx = -1

        while idx < N:
            item, dataset_idx = test_dataset[idx]

            if dataset_idx > prev_dataset_idx:
                model.reset_states()

            events = item[0]['event'].unsqueeze(dim=0)
            frame  = item[0]['frame'].unsqueeze(dim=0)
            target = item[0]['depth']

            events = events.float().to(gpu)
            frame = frame.float().to(gpu)

            pred_dict = model(events, frame)
            pred_depth = pred_dict['pred_depth']
            pred_depth = disp_to_depth(pred_depth)

            if len(pred_depth.shape) > 3:
                pred_depth = pred_depth.squeeze(dim=0).cpu().numpy()

            if target is not None:
                target = target.cpu().numpy()
                np.save(join(pred_depth_dir, '{:06d}.npy'.format(idx)), pred_depth)
                np.save(join(gt_dir, '{:06d}.npy'.format(idx)), target)

            # save event_image
            input_dir_events = join(inputs_dir, "events")
            ensure_dir(input_dir_events)
            events = item[0]['event'].cpu().numpy()
            events_data = np.sum(events, axis=0)
            negativ_input = np.where(events_data <= -0.5, 1.0, 0.0)
            positiv_input = np.where(events_data > 0.9, 1.0, 0.0)
            zeros_input = np.zeros_like(events_data)
            total_image = np.concatenate(
                (negativ_input[:, :, None], zeros_input[:, :, None], positiv_input[:, :, None]), axis=2)
            mask = (negativ_input + positiv_input).astype(np.bool_)
            img = np.ones_like(total_image)
            img[mask] = total_image[mask]
            cv2.imwrite(join(input_dir_events, '{:06d}.png'.format(idx)),
                        img * 255.0)

            # save frame_image
            input_dir_frame = join(inputs_dir, "frame")
            ensure_dir(input_dir_frame)
            frame = item[0]['frame'].cpu().numpy()
            frame_data = np.sum(frame, axis=0) / 3
            cv2.imwrite(join(input_dir_frame, '{:06d}.png'.format(idx)),
                        frame_data * 255.0)

            # save pred color image
            color_map_dir_pred = join(color_map_dir, "pred")
            ensure_dir(color_map_dir_pred)
            color_map = make_colormap(pred_depth, color_mapper_overall)
            cv2.imwrite(join(color_map_dir_pred, '{:06d}.png'.format(idx)), color_map * 255.0)

            # save gt color image
            color_map_dir_gt = join(color_map_dir, "gt")
            ensure_dir(color_map_dir_gt)
            color_map = make_colormap(target, color_mapper_overall)
            cv2.imwrite(join(color_map_dir_gt, '{:06d}.png'.format(idx)), color_map * 255.0)

            print(idx)
            idx += 1
            prev_dataset_idx = dataset_idx


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(
        description='Learning DVS Image Reconstruction')
    parser.add_argument('--path_to_model', type=str,
                        help='path to the model weights',
                        default='./save_checkpoint/CFRNet_DSEC_epoch20_bs_3_lr_00032/model_best.pth.tar')
    parser.add_argument('--output_path', type=str,
                        help='path to folder for saving outputs',
                        default='./save_checkpoint/CFRNet_DSEC_epoch20_bs_3_lr_00032/test_out')
    parser.add_argument('--config', type=str,
                        help='path to config. If not specified, config from model folder is taken',
                        default=None)
    parser.add_argument('--data_folder', type=str,
                        help='path to folder of data to be tested',
                        default=None)

    args = parser.parse_args()

    if args.config is None:
        head_tail = os.path.split(args.path_to_model)
        config = json.load(open(os.path.join(head_tail[0], 'config.json')))
    else:
        config = json.load(open(args.config))

    main(config, args.path_to_model, args.output_path)