import os
from pathlib import Path
import weakref
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as f
import glob
from numpy.lib.format import open_memmap

import cv2
from omegaconf import OmegaConf
import h5py
import random
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from utils.representations import VoxelGrid
from utils.eventslicer import EventSlicer
from utils.event_tensor_utils import events_to_voxel_grid, make_color_histo


class Sequence(Dataset):

    def __init__(self, base_folder, root_folder, mode='train', delta_t_ms=50, sequence_length=2, transform=None,
                 step_size=20, clip_distance=100.0, num_bins=15, normalize=True):

        assert(sequence_length > 0)
        assert(step_size > 0)
        assert(clip_distance > 0)
        self.L = sequence_length

        self.dataset = DSEC_depth(base_folder, root_folder, clip_distance=clip_distance, mode=mode, delta_t_ms=delta_t_ms,
                                 num_bins=num_bins, transform=transform, normalize=normalize)
        self.step_size = step_size
        if self.L >= self.dataset.length:
            self.length = 0
        else:
            self.length = (self.dataset.length - self.L) // self.step_size + 1
        print("lenght sequence dataset: ", self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, i):

        assert(i >= 0)
        assert(i < self.length)

        # generate a random seed here, that we will pass to the transform function
        # of each item, to make sure all the items in the sequence are transformed
        # in the same way
        seed = random.randint(0, 2**32)

        sequence = []

        # add the first element (i.e. do not start with a pause)
        k = 0
        j = i * self.step_size
        item = self.dataset.__getitem__(j, seed)
        sequence.append(item)

        for n in range(self.L - 1):
            k += 1
            item = self.dataset.__getitem__(j + k, seed)
            sequence.append(item)
        #print(sum(len(k) for k in sequence))

        return sequence


class DSEC_depth(Dataset):
    def __init__(self, base_folder,
                 root_folder,
                 clip_distance=54.0,
                 mode: str = 'train',
                 delta_t_ms: int = 50,
                 num_bins: int = 15,
                 transform=None,
                 normalize=True):

        self.base_folder = base_folder
        self.root_folder = Path(root_folder)
        self.transform = transform
        self.normalize = normalize
        self.eps = 1e-05
        self.clip_distance = clip_distance
        self.fill_out_nans = False
        self.mode = mode

        if self.mode == 'train':
            seq_path = self.root_folder / 'train_sequence'
        elif self.mode == 'validation':
            seq_path = self.root_folder / 'test_sequence'

        seq_path = seq_path / self.base_folder

        assert num_bins >= 1
        assert delta_t_ms <= 100, 'adapt this code, if duration is higher than 100 ms'
        assert seq_path.is_dir()

        # Save output dimensions
        self.height = 480
        self.width = 640
        self.num_bins = num_bins

        # Set event representation
        self.voxel_grid = VoxelGrid(self.num_bins, 480, 640, normalize=self.normalize)

        # Save delta timestamp in ms
        self.delta_t_us = delta_t_ms * 1000

        # load disparity timestamps
        disp_dir = seq_path / 'disparity_event'
        assert disp_dir.is_dir()
        self.timestamps = np.loadtxt(seq_path / 'disparity_timestamps.txt', dtype='int64')

        # load disparity paths
        disp_gt_pathstrings = list()
        for entry in disp_dir.iterdir():
            assert str(entry.name).endswith('.png')
            disp_gt_pathstrings.append(str(entry))
        disp_gt_pathstrings.sort()
        self.disp_gt_pathstrings = disp_gt_pathstrings
        self.image_dir = seq_path / 'images_rectified_aligned'

        confpath = self.root_folder / 'train_calibration' / self.base_folder / 'calibration' / 'cam_to_cam.yaml'
        assert confpath.exists()
        self.conf = OmegaConf.load(confpath)

        assert len(self.disp_gt_pathstrings) == self.timestamps.size

        # Remove first disparity path and corresponding timestamp.
        # This is necessary because we do not have events before the first disparity map.
        assert int(Path(self.disp_gt_pathstrings[0]).stem) == 0
        self.disp_gt_pathstrings.pop(0)
        self.timestamps = self.timestamps[1:]

        ev_dir = seq_path / 'events'
        ev_data_file = ev_dir / 'events.h5'
        ev_rect_file = ev_dir / 'rectify_map.h5'
        h5f = h5py.File(str(ev_data_file), 'r')
        self.h5f = h5f
        self.event_slicers = EventSlicer(h5f)
        with h5py.File(str(ev_rect_file), 'r') as h5_rect:
            self.rectify_ev_maps = h5_rect['rectify_map'][()]

        self.length = len(self.disp_gt_pathstrings)

        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)

    def events_to_voxel_grid(self, x, y, p, t, device: str='cpu'):
        t = (t - t[0]).astype('float32')
        t = (t/t[-1])
        x = x.astype('float32')
        y = y.astype('float32')
        pol = p.astype('float32')
        return self.voxel_grid.convert(
                torch.from_numpy(x),
                torch.from_numpy(y),
                torch.from_numpy(pol),
                torch.from_numpy(t))

    def __len__(self):
        return self.length

    def get_disparity_map(self, filepath: Path):
        assert filepath.is_file()
        disp_16bit = cv2.imread(str(filepath), cv2.IMREAD_ANYDEPTH)
        disp = disp_16bit.astype('float32') / 256
        # disp = disp_16bit.astype('float32')
        if len(disp.shape) > 2:
            if disp.shape[2] > 1:
                disp = self.rgb2gray(disp)  # [H x W]
        return disp

    def rectify_events(self, x: np.ndarray, y: np.ndarray):
        # From distorted to undistorted
        rectify_map = self.rectify_ev_maps
        assert rectify_map.shape == (480, 640, 2), rectify_map.shape
        assert x.max() < 640
        assert y.max() < 480
        return rectify_map[y, x]

    def disparity_to_depth(self, disp, baseline, focal):
        # convert disparity to depth map: z = b*f / d
        depth = disp
        depth[disp > 0] = (baseline * focal) / disp[disp > 0]
        return depth

    @staticmethod
    def close_callback(h5f_dict):
        # for k, h5f in h5f_dict.items():
        h5f_dict.close()

    def file_filter(self, file):
        if file[-4:] in ['.jpg', '.png', '.bmp']:
            return True
        else:
            return False

    def prepare_depth(self, depth, seed):

        if len(depth.shape) == 2:  # [H x W] grayscale image -> [H x W x 1]
            depth = np.expand_dims(depth, -1)

        depth = np.moveaxis(depth, -1, 0)  # H x W x C -> C x H x W
        depth = torch.from_numpy(depth)  # numpy to tensor

        if self.transform:
            random.seed(seed)
            depth = self.transform(depth)

        return depth

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1141]).astype(np.float32)

    def prepare_frame(self, frame, seed):
        if len(frame.shape) > 2:
            if frame.shape[2] > 1:
                frame = self.rgb2gray(frame)  # [H x W]
        frame = frame.astype(np.float32)

        frame /= 255.0  # normalize
        # frame = np.expand_dims(frame, axis=0)  # expand to [1 x H x W]
        frame = np.stack((frame,) * 3, axis=0) # [3 x H x W]
        frame = torch.from_numpy(frame)
        if self.transform:
            random.seed(seed)
            frame = self.transform(frame)

        return frame

    def __getitem__(self, index, seed=None, reg_factor=2.38597):

        assert (index >= 0)

        if seed is None:
            # if no specific random seed was passed, generate our own.
            # otherwise, use the seed that was passed to us
            seed = random.randint(0, 2 ** 32)

        ts_end = self.timestamps[index]
        # ts_start should be fine (within the window as we removed the first disparity map)
        ts_start = ts_end - self.delta_t_us

        disp_gt_path = Path(self.disp_gt_pathstrings[index])
        file_index = int(disp_gt_path.stem)

        # load depth_gt
        disparity_gt = self.get_disparity_map(disp_gt_path)
        cams_img = self.conf['disparity_to_depth']['cams_03']
        baseline = 1.0 / cams_img[3][2]
        focal = cams_img[2][3]
        depth_gt = self.disparity_to_depth(disparity_gt, baseline, focal)
        # Crop out the region corresponding to the invalid edge of the ground truth depth map
        depth_gt = depth_gt[:416, 48:]
        depth_gt = self.prepare_depth(depth_gt, seed)

        # load event
        event_data = self.event_slicers.get_events(ts_start, ts_end)
        p = event_data['p']
        t = event_data['t']
        x = event_data['x']
        y = event_data['y']
        xy_rect = self.rectify_events(x, y)
        x_rect = xy_rect[:, 0]
        y_rect = xy_rect[:, 1]
        event = self.events_to_voxel_grid(x_rect, y_rect, p, t)
        event = event[:, :416, 48:]
        if self.transform:
            random.seed(seed)
            event = self.transform(event)

        # load frame  The rgb frame rate is 1/4 of the event frame rate
        img_num = (((file_index / 2 - 1) // 4) * 4 + 1) * 2
        file = "{:06}".format(int(img_num)) + ".png"
        img_path = self.image_dir / file
        frame = cv2.imread(str(img_path)).astype(np.float32)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        frame = frame[:416, 48:, :]
        frame = self.prepare_frame(frame, seed)

        item = {'event': event,
                'frame': frame,
                'depth': depth_gt}

        return item

# Loading data with high frame rate at 100Hz
class DSEC_HFR(Dataset):
    def __init__(self, base_folder,
                 root_folder,
                 clip_distance=54.0,
                 mode: str = 'train',
                 delta_t_ms: int = 50,
                 num_bins: int = 15,
                 transform=None,
                 normalize=True):

        self.base_folder = base_folder
        self.root_folder = Path(root_folder)
        self.transform = transform
        self.normalize = normalize
        self.eps = 1e-05
        self.clip_distance = clip_distance
        self.fill_out_nans = False
        self.mode = mode

        if self.mode == 'train':
            seq_path = self.root_folder / 'train_sequence'
        elif self.mode == 'validation':
            seq_path = self.root_folder / 'test_sequence'

        seq_path = seq_path / self.base_folder

        assert num_bins >= 1
        assert delta_t_ms <= 100, 'adapt this code, if duration is higher than 100 ms'
        assert seq_path.is_dir()

        # Save output dimensions
        self.height = 480
        self.width = 640
        self.num_bins = num_bins

        # Set event representation
        self.voxel_grid = VoxelGrid(self.num_bins, 480, 640, normalize=self.normalize)

        # Save delta timestamp in ms
        self.delta_t_us = delta_t_ms * 1000

        # load disparity timestamps
        disp_dir = seq_path / 'disparity_event'
        assert disp_dir.is_dir()
        self.timestamps = np.loadtxt(seq_path / 'disparity_timestamps.txt', dtype='int64')

        # load disparity paths
        disp_gt_pathstrings = list()
        for entry in disp_dir.iterdir():
            assert str(entry.name).endswith('.png')
            disp_gt_pathstrings.append(str(entry))
        disp_gt_pathstrings.sort()
        self.disp_gt_pathstrings = disp_gt_pathstrings
        self.image_dir = seq_path / 'images_rectified_aligned'

        confpath = self.root_folder / 'train_calibration' / self.base_folder / 'calibration' / 'cam_to_cam.yaml'
        assert confpath.exists()
        self.conf = OmegaConf.load(confpath)

        assert len(self.disp_gt_pathstrings) == self.timestamps.size

        # Remove first disparity path and corresponding timestamp.
        # This is necessary because we do not have events before the first disparity map.
        assert int(Path(self.disp_gt_pathstrings[0]).stem) == 0
        self.disp_gt_pathstrings.pop(0)
        self.timestamps = self.timestamps[1:]

        ev_dir = seq_path / 'events'
        ev_data_file = ev_dir / 'events.h5'
        ev_rect_file = ev_dir / 'rectify_map.h5'
        h5f = h5py.File(str(ev_data_file), 'r')
        self.h5f = h5f
        self.event_slicers = EventSlicer(h5f)
        with h5py.File(str(ev_rect_file), 'r') as h5_rect:
            self.rectify_ev_maps = h5_rect['rectify_map'][()]

        self.length = (len(self.disp_gt_pathstrings) - 1) * 10 + 1

        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)

    def events_to_voxel_grid(self, x, y, p, t, device: str='cpu'):
        t = (t - t[0]).astype('float32')
        t = (t/t[-1])
        x = x.astype('float32')
        y = y.astype('float32')
        pol = p.astype('float32')
        return self.voxel_grid.convert(
                torch.from_numpy(x),
                torch.from_numpy(y),
                torch.from_numpy(pol),
                torch.from_numpy(t))

    def __len__(self):
        return self.length

    def get_disparity_map(self, filepath: Path):
        assert filepath.is_file()
        disp_16bit = cv2.imread(str(filepath), cv2.IMREAD_ANYDEPTH)
        disp = disp_16bit.astype('float32') / 256
        # disp = disp_16bit.astype('float32')
        if len(disp.shape) > 2:
            if disp.shape[2] > 1:
                disp = self.rgb2gray(disp)  # [H x W]
        return disp

    def rectify_events(self, x: np.ndarray, y: np.ndarray):
        # From distorted to undistorted
        rectify_map = self.rectify_ev_maps
        assert rectify_map.shape == (480, 640, 2), rectify_map.shape
        assert x.max() < 640
        assert y.max() < 480
        return rectify_map[y, x]

    def disparity_to_depth(self, disp, baseline, focal):
        # convert disparity to depth map: z = b*f / d
        depth = disp
        depth[disp > 0] = (baseline * focal) / disp[disp > 0]
        return depth

    @staticmethod
    def close_callback(h5f_dict):
        # for k, h5f in h5f_dict.items():
        h5f_dict.close()

    def file_filter(self, file):
        if file[-4:] in ['.jpg', '.png', '.bmp']:
            return True
        else:
            return False

    def prepare_depth(self, depth, seed):

        if len(depth.shape) == 2:  # [H x W] grayscale image -> [H x W x 1]
            depth = np.expand_dims(depth, -1)

        depth = np.moveaxis(depth, -1, 0)  # H x W x C -> C x H x W
        depth = torch.from_numpy(depth)  # numpy to tensor

        if self.transform:
            random.seed(seed)
            depth = self.transform(depth)

        return depth

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1141]).astype(np.float32)

    def prepare_frame(self, frame, seed):
        if len(frame.shape) > 2:
            if frame.shape[2] > 1:
                frame = self.rgb2gray(frame)  # [H x W]
        frame = frame.astype(np.float32)

        frame /= 255.0  # normalize
        # frame = np.expand_dims(frame, axis=0)  # expand to [1 x H x W]
        frame = np.stack((frame,) * 3, axis=0) # [3 x H x W]
        frame = torch.from_numpy(frame)
        if self.transform:
            random.seed(seed)
            frame = self.transform(frame)

        return frame

    def __getitem__(self, index, seed=None, reg_factor=2.38597):

        assert (index >= 0)

        if seed is None:
            # if no specific random seed was passed, generate our own.
            # otherwise, use the seed that was passed to us
            seed = random.randint(0, 2 ** 32)

        depth_delta_t_ms = 100
        event_delta_t_ms = 10
        scale = depth_delta_t_ms // event_delta_t_ms
        depth_index = index // scale
        num = index % scale

        ts_end = self.timestamps[depth_index] + num * event_delta_t_ms * 1000
        # ts_start should be fine (within the window as we removed the first disparity map)
        ts_start = ts_end - self.delta_t_us

        disp_gt_path = Path(self.disp_gt_pathstrings[depth_index])
        file_index = int(disp_gt_path.stem)

        if num == 0:
            # load depth_gt
            disparity_gt = self.get_disparity_map(disp_gt_path)
            cams_img = self.conf['disparity_to_depth']['cams_03']
            baseline = 1.0 / cams_img[3][2]
            focal = cams_img[2][3]
            depth_gt = self.disparity_to_depth(disparity_gt, baseline, focal)
            depth_gt = depth_gt[:416, 48:]
            depth_gt = self.prepare_depth(depth_gt, seed)
        else:
            depth_gt = None

        # load event
        event_data = self.event_slicers.get_events(ts_start, ts_end)
        p = event_data['p']
        t = event_data['t']
        x = event_data['x']
        y = event_data['y']
        xy_rect = self.rectify_events(x, y)
        x_rect = xy_rect[:, 0]
        y_rect = xy_rect[:, 1]
        event = self.events_to_voxel_grid(x_rect, y_rect, p, t)
        event = event[:, :416, 48:]
        if self.transform:
            random.seed(seed)
            event = self.transform(event)

        # load frame 2.5 hz
        img_num = (((file_index / 2 - 1) // 4) * 4 + 1) * 2
        # load frame 10 hz
        # img_num = file_index
        file = "{:06}".format(int(img_num)) + ".png"
        img_path = self.image_dir / file
        frame = cv2.imread(str(img_path)).astype(np.float32)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        frame = frame[:416, 48:, :]
        frame = self.prepare_frame(frame, seed)

        item = {'event': event,
                'frame': frame,
                'depth': depth_gt}

        return item

class MVSEC_data(Dataset):
    def __init__(self, base_folder,
                 root_folder,
                 clip_distance=80.0,
                 mode: str = 'train',
                 delta_t_ms: int = 50,
                 num_bins: int = 3,
                 transform=None,
                 normalize=True):

        self.base_folder = base_folder
        self.root_folder = Path(root_folder)
        self.transform = transform
        self.normalize = normalize
        self.clip_distance = clip_distance
        self.mode = mode
        self.eps = 1e-05

        if self.mode == 'train':
            seq_path = self.root_folder / 'train'
        elif self.mode == 'validation':
            seq_path = self.root_folder / 'validation'

        seq_path = seq_path / self.base_folder

        assert num_bins >= 1
        assert seq_path.is_dir()

        # Save output dimensions
        self.height = 260
        self.width = 346
        self.num_bins = num_bins

        self.target_label = "depths"
        self.events_label = "events"
        self.frame_label = "frames"
        self.depth_to_event_index = np.genfromtxt(os.path.join(str(seq_path), "depth_to_event_index.txt"), dtype="int32")
        self.depth_to_frame_index = np.genfromtxt(os.path.join(str(seq_path), "depth_to_frame_index.txt"), dtype="int32")

        self.depth_paths = sorted(glob.glob(os.path.join(str(seq_path), self.target_label, "*.npy")))
        self.frame_paths = sorted(glob.glob(os.path.join(str(seq_path), self.frame_label, "*.png")))
        self.event_handles = [
            open_memmap(os.path.join(str(seq_path), self.events_label, "t.npy"), mode='r'),
            open_memmap(os.path.join(str(seq_path), self.events_label, "x.npy"), mode='r'),
            open_memmap(os.path.join(str(seq_path), self.events_label, "y.npy"), mode='r'),
            open_memmap(os.path.join(str(seq_path), self.events_label, "p.npy"), mode='r')
        ]

        self.length = len(self.depth_to_event_index)-1

    def __len__(self):
        return self.length

    def prepare_depth(self, depth, seed):

        if len(depth.shape) == 2:  # [H x W] grayscale image -> [H x W x 1]
            depth = np.expand_dims(depth, -1)

        depth = np.moveaxis(depth, -1, 0)  # H x W x C -> C x H x W
        depth = torch.from_numpy(depth)  # numpy to tensor

        if self.transform:
            random.seed(seed)
            depth = self.transform(depth)

        return depth

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1141]).astype(np.float32)

    def prepare_frame(self, frame, seed):
        if len(frame.shape) > 2:
            if frame.shape[2] > 1:
                frame = self.rgb2gray(frame)  # [H x W]
        frame = frame.astype(np.float32)

        frame /= 255.0  # normalize
        # frame = np.expand_dims(frame, axis=0)  # expand to [1 x H x W]
        frame = np.stack((frame,) * 3, axis=0) # [3 x H x W]
        frame = torch.from_numpy(frame)
        if self.transform:
            random.seed(seed)
            frame = self.transform(frame)

        return frame

    def normalize_voxelgrid(self, event_tensor):
        # normalize the event tensor (voxel grid) in such a way that the mean and stddev of the nonzero values
        # in the tensor are equal to (0.0, 1.0)
        mask = np.nonzero(event_tensor)
        if mask[0].size > 0:
            mean, stddev = event_tensor[mask].mean(), event_tensor[mask].std()
            if stddev > 0:
                event_tensor[mask] = (event_tensor[mask] - mean) / stddev
        return event_tensor

    def prepare_event(self, event, seed, normalize=False):

        if normalize:
            event = self.normalize_voxelgrid(event)
        event = torch.from_numpy(event)

        if self.transform:
            random.seed(seed)
            event = self.transform(event)

        return event

    def __getitem__(self, index, seed=None):

        assert (index >= 0)

        if seed is None:
            seed = random.randint(0, 2 ** 32)

        # events
        event_t, event_x, event_y, event_p = self.event_handles

        event_t = np.expand_dims(event_t, axis=1)
        event_x = np.expand_dims(event_x, axis=1)
        event_y = np.expand_dims(event_y, axis=1)
        event_p = np.expand_dims(event_p, axis=1)

        event_max_index = self.depth_to_event_index[index + 1]
        event_min_index = self.depth_to_event_index[index]
        events = np.concatenate([
            event_t[event_min_index:event_max_index, :].astype("float64"),
            event_x[event_min_index:event_max_index, :].astype("float32"),
            event_y[event_min_index:event_max_index, :].astype("float32"),
            (event_p[event_min_index:event_max_index, :].astype("float32"))
        ], axis=-1)
        event = events_to_voxel_grid(events, self.num_bins, self.width, self.height)

        event = event[:, 1:257, 5:341]
        event = self.prepare_event(event, seed, normalize=self.normalize)


        # frames
        frame_index = self.depth_to_frame_index[index + 1]
        frame = cv2.imread(self.frame_paths[frame_index]).astype(np.float32)
        frame = frame[1:257, 5:341, :]
        frame = self.prepare_frame(frame, seed)

        # depths
        depth_gt = np.load(os.path.join(self.depth_paths[index + 1])).astype(np.float32)
        depth_gt = depth_gt[1:257, 5:341]
        depth_gt = self.prepare_depth(depth_gt, seed)

        item = {'frame': frame,
                'frame': frame,
                'depth': depth_gt}

        return item



def collate_mvsec(batch):
    return_sequence = []
    sequence_length = len(batch[0])
    batch_size = len(batch)
    for j in range(sequence_length):
        # loop over the whole sequence to fill return_sequence list
        return_sequence += [[batch[i][j] for i in range(batch_size)]]
    return return_sequence

