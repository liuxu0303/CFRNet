import os
from os.path import join
import numpy as np
import json
import logging
import argparse
import torch
from model.model import *
from model.loss import *
from torch.utils.data import DataLoader, ConcatDataset
# from data_loader.dataset import *
from data_loader.data_loaders import HDF5DataLoader
from trainer.trainer import Trainer
from utils.data_augmentation import Compose, RandomRotationFlip, RandomCrop, CenterCrop, BottomRightCrop
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='')


def main(config, backbone_pretrain=None, initial_checkpoint=None):
    train_logger = None

    L = config['trainer']['sequence_length']
    assert (L > 0)

    dataset_type, base_folder, root_folder = {}, {}, {}

    step_size = {}
    delta_t_ms = {}
    clip_distance = {}
    num_bins = {}

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

    np.random.seed(0)

    normalize = config['data_loader'].get('normalize', True)

    data_loader = eval(dataset_type['train'])(data_file=base_folder['train'],
                                              batch_size=config['data_loader']['batch_size'],
                                              shuffle=config['data_loader']['shuffle'],
                                              num_workers=config['data_loader']['num_workers'],
                                              pin_memory=config['data_loader']['pin_memory'],
                                              sequence_kwargs={'root_folder': root_folder['train'],
                                                               'mode': 'train',
                                                               'delta_t_ms': delta_t_ms['train'],
                                                               'sequence_length': L,
                                                               'transform': Compose([RandomRotationFlip(0.0, 0.5, 0.0)]),
                                                               'step_size': step_size['train'],
                                                               'clip_distance': clip_distance['train'],
                                                               'num_bins': num_bins['train'],
                                                               'normalize': normalize})

    valid_data_loader = eval(dataset_type['validation'])(data_file=base_folder['validation'],
                                                         batch_size=config['data_loader']['batch_size'],
                                                         shuffle=False,
                                                         num_workers=config['data_loader']['num_workers'],
                                                         pin_memory=config['data_loader']['pin_memory'],
                                                         sequence_kwargs={'root_folder': root_folder['validation'],
                                                                          'mode': 'validation',
                                                                          'delta_t_ms': delta_t_ms['validation'],
                                                                          'sequence_length': L,
                                                                          'transform': None,
                                                                          'step_size': step_size['validation'],
                                                                          'clip_distance': clip_distance['validation'],
                                                                          'num_bins': num_bins['validation'],
                                                                          'normalize': normalize})

    torch.manual_seed(0)
    model = eval(config['arch'])(config['model'])
    # Initialize weight randomly
    model.init_weights()

    if initial_checkpoint is not None:
        print('Loading initial model weights from: {}'.format(initial_checkpoint))
        checkpoint = torch.load(initial_checkpoint)
        model.load_state_dict(checkpoint['state_dict'])


    # Initialize the modality-specific shared encoders with pretrained lite-mono
    if backbone_pretrain is not None:
        print('Loading pretrained model weights from: {}'.format(backbone_pretrain))
        pretrained_dict = torch.load(backbone_pretrain)['model']
        rgb_model_dict = model.Unet.RGB_encoder.state_dict()
        event_model_dict = model.Unet.event_encoder.state_dict()
        rgb_pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               (k in rgb_model_dict and not k.startswith('norm'))}
        event_pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               (k in event_model_dict and not k.startswith('norm'))}
        rgb_model_dict.update(rgb_pretrained_dict)
        event_model_dict.update(event_pretrained_dict)
        model.Unet.RGB_encoder.load_state_dict(rgb_model_dict)
        model.Unet.event_encoder.load_state_dict(event_model_dict)
        print('my pretrain loaded.')

    loss = eval(config['loss']['type'])
    loss_params = config['loss']['config'] if 'config' in config['loss'] else None
    print("Using %s with config %s" % (config['loss']['type'], config['loss']['config']))

    trainer = Trainer(model, loss, loss_params,
              config=config,
              data_loader=data_loader,
              valid_data_loader=valid_data_loader,
              train_logger=train_logger)

    trainer.train()


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(
        description='Learning DVS Monocular Depth Prediction')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-b', '--backbone_pretrain',
                        default="./lite_transformer/lite-mono-pretrain.pth",
                        type=str,
                        help='path to the checkpoint with which to initialize the modality-specific shared encoders.')
    parser.add_argument('--c', type=str,
                        help='path to the model weights',
                        default=None)
    parser.add_argument('-g', '--gpu_id', default=None, type=int,
                        help='path to the checkpoint with which to initialize the model weights (default: None)')
    args = parser.parse_args()

    if args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    config = None
    if args.config is not None:
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
        if args.resume is None:
            assert not os.path.exists(path), "Path {} already exists!".format(path)
    assert config is not None

    main(config, args.backbone_pretrain, args.path_to_model)
