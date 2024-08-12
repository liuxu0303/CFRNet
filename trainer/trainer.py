import torch
import torch.nn as nn
from base import BaseTrainer

class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
        self.optimizer is by default handled by BaseTrainer based on config.
    """

    def __init__(self, model, loss, loss_params, config,
                 data_loader, valid_data_loader=None, resume=None, train_logger=None):
        super(Trainer, self).__init__(model, loss, loss_params, config, resume, train_logger)
        self.config = config
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.valid = True if self.valid_data_loader is not None else False
        self.log_step = 1
        self.added_tensorboard_graph = False
        self.max_depth = config['data_loader']['train']['clip_distance']
        self.min_depth = loss_params['min_depth']


    def calculate_total_batch_loss(self, loss_dict, total_loss_dict, L=1):
        nominal_loss = sum(loss_dict['losses']) / float(L)

        losses = []
        losses.append(nominal_loss)

        loss = sum(losses)

        # add all losses in a dict for logging
        with torch.no_grad():
            if not total_loss_dict:
                total_loss_dict = {'loss': loss, 'L_si': nominal_loss}

            else:
                total_loss_dict['loss'] += loss
                total_loss_dict['L_si'] += nominal_loss

        return total_loss_dict

    def disp_to_depth(self, disp):
        min_disp = 1 / self.max_depth
        max_disp = 1 / self.min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1 / scaled_disp
        return depth

    def forward_pass_sequence(self, sequence):
        # 'sequence' is a list containing L successive events, frames <-> depths pairs
        # each element in 'sequence' is a dictionary containing the keys 'event', 'frame' and 'depth'
        L = len(sequence)
        assert (L > 0)

        total_batch_losses = {}

        loss_dict = {'losses': []}

        self.model.reset_states()
        for i, batch_item in enumerate(sequence):
            events = batch_item['event']
            frame = batch_item['frame']
            target = batch_item['depth']

            events = events.float().to(self.gpu)
            frame = frame.float().to(self.gpu)
            target = target.float().to(self.gpu)

            pred_dict = self.model(events, frame)
            pred_depth = pred_dict['pred_depth']

            pred_depth = self.disp_to_depth(pred_depth)

            # calculate loss
            if self.loss_params is not None:
                loss_dict['losses'].append(
                    self.loss(pred_depth, target, **self.loss_params))
            else:
                loss_dict['losses'].append(self.loss(pred_depth, target))

        total_batch_losses = self.calculate_total_batch_loss(loss_dict, total_batch_losses, L)

        return total_batch_losses

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        """

        self.model.train()

        all_losses_in_batch = {}
        for batch_idx, sequence in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            losses = self.forward_pass_sequence(sequence)
            loss = losses['loss']
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.2)
            self.optimizer.step()
            self.lr_scheduler.step()

            with torch.no_grad():
                for loss_name, loss_value in losses.items():
                    if loss_name not in all_losses_in_batch:
                        all_losses_in_batch[loss_name] = []
                    all_losses_in_batch[loss_name].append(loss_value.item())

                if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                    loss_str = ''
                    for loss_name, loss_value in losses.items():
                        loss_str += '{}: {:.4f} '.format(loss_name, loss_value.item())
                    self.logger.info('Train Epoch: {}, batch_idx: {}, [{}/{} ({:.0f}%)] {}'.format(
                        epoch, batch_idx,
                        batch_idx * self.data_loader.batch_size,
                        len(self.data_loader) * self.data_loader.batch_size,
                        100.0 * batch_idx / len(self.data_loader),
                        loss_str))

        # compute average losses over the batch
        total_losses = {loss_name: sum(loss_values) / len(self.data_loader)
                        for loss_name, loss_values in all_losses_in_batch.items()}
        log = {
            'loss': total_losses['loss'],
            'losses': total_losses
        }

        if self.valid:
            val_log = self._valid_epoch(epoch=epoch)
            log = {**log, **val_log}

        return log

    def _valid_epoch(self, epoch=0):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        all_losses_in_batch = {}
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.valid_data_loader):
                losses = self.forward_pass_sequence(batch)
                for loss_name, loss_value in losses.items():
                    if loss_name not in all_losses_in_batch:
                        all_losses_in_batch[loss_name] = []
                    all_losses_in_batch[loss_name].append(loss_value.item())

                if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                    self.logger.info('Validation: [{}/{} ({:.0f}%)]'.format(
                        batch_idx * self.valid_data_loader.batch_size,
                        len(self.valid_data_loader) * self.valid_data_loader.batch_size,
                        100.0 * batch_idx / len(self.valid_data_loader)))

        total_losses = {loss_name: sum(loss_values) / len(self.valid_data_loader)
                        for loss_name, loss_values in all_losses_in_batch.items()}

        return {'val_L_si': total_losses['L_si'],
                'val_losses': total_losses}
