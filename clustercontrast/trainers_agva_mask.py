"""Baseline trainers that suppress augmented-sample memory updates only."""

from __future__ import absolute_import, print_function

import time

import torch

from .trainers import ClusterContrastTrainer_DCL
from .trainers_no_ema_loss import (
    ClusterContrastTrainer_PCLMP_NoEMALoss,
    compose_primary_loss_without_ema,
)
from .utils.meters import AverageMeter


class _AGVAMemoryMaskStats(object):
    def _init_agva_stats(self, mask_memory_updates):
        self.mask_memory_updates = bool(mask_memory_updates)
        self._agva_samples = 0
        self._agva_augmented = 0
        self._agva_skipped = 0
        self._agva_classes_unchanged = 0

    def _record_aerial_batch(self, is_aug):
        is_aug = torch.as_tensor(is_aug, dtype=torch.bool)
        self._agva_samples += int(is_aug.numel())
        self._agva_augmented += int(is_aug.sum().item())
        return ~is_aug.cuda()

    def _aerial_memory(self, memory, features, targets, update_mask,
                       model_name='encoder'):
        if not self.mask_memory_updates:
            return memory(features, targets, model_name=model_name)
        update_mask = update_mask.to(device=targets.device, dtype=torch.bool)
        self._agva_skipped += int((~update_mask).sum().item())
        for label in torch.unique(targets):
            members = targets == label
            if not bool(update_mask[members].any()):
                self._agva_classes_unchanged += 1
        return memory(features, targets, model_name=model_name,
                      update_mask=update_mask)

    def _print_agva_stats(self, epoch):
        ratio = (self._agva_augmented / float(self._agva_samples)
                 if self._agva_samples else 0.0)
        print('[AGVA-MM] epoch {}: aerial samples {}, augmented {} '
              '({:.2%}), skipped updates {}, classes kept unchanged {}'
              .format(epoch, self._agva_samples, self._agva_augmented,
                      ratio, self._agva_skipped,
                      self._agva_classes_unchanged))


class ClusterContrastTrainer_DCL_AGVAMask(
        _AGVAMemoryMaskStats, ClusterContrastTrainer_DCL):
    def __init__(self, encoder, memory=None, debug_nonfinite=False,
                 mask_memory_updates=True):
        super(ClusterContrastTrainer_DCL_AGVAMask, self).__init__(
            encoder, memory=memory, debug_nonfinite=debug_nonfinite)
        self._init_agva_stats(mask_memory_updates)

    @staticmethod
    def _parse_data_ir_flag(inputs):
        imgs, _, pids, _, indexes, is_aug = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda(), is_aug

    def train(self, epoch, data_loader_ir, data_loader_rgb, optimizer,
              print_freq=10, train_iters=400):
        self.encoder.train()
        self._init_agva_stats(self.mask_memory_updates)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()
        for i in range(train_iters):
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)
            inputs_ir, labels_ir, indexes_ir, is_aug = \
                self._parse_data_ir_flag(inputs_ir)
            update_mask_ir = self._record_aerial_batch(is_aug)
            inputs_rgb, inputs_rgb1, labels_rgb, indexes_rgb = \
                self._parse_data_rgb(inputs_rgb)
            iteration = i + 1
            self._check_tensor('inputs_ir', inputs_ir, epoch, iteration)
            self._check_tensor('inputs_rgb_view1', inputs_rgb, epoch, iteration)
            self._check_tensor('inputs_rgb_view2', inputs_rgb1, epoch, iteration)
            inputs_rgb = torch.cat((inputs_rgb, inputs_rgb1), 0)
            labels_rgb = torch.cat((labels_rgb, labels_rgb), -1)
            (_, f_out_rgb, f_out_ir, labels_rgb, labels_ir,
             pool_rgb, pool_ir) = self._forward(
                 inputs_rgb, inputs_ir, label_1=labels_rgb,
                 label_2=labels_ir, modal=0)
            self._check_tensor('features_ir', f_out_ir, epoch, iteration)
            self._check_tensor('features_rgb', f_out_rgb, epoch, iteration)
            self._check_tensor('pool_ir', pool_ir, epoch, iteration)
            self._check_tensor('pool_rgb', pool_rgb, epoch, iteration)
            self._check_tensor('memory_ir.features', self.memory_ir.features,
                               epoch, iteration)
            self._check_tensor('memory_rgb.features', self.memory_rgb.features,
                               epoch, iteration)
            loss_ir = self._aerial_memory(
                self.memory_ir, f_out_ir, labels_ir, update_mask_ir)
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)
            loss = loss_ir + loss_rgb
            self._check_tensor('loss_ir', loss_ir, epoch, iteration)
            self._check_tensor('loss_rgb', loss_rgb, epoch, iteration)
            self._check_tensor('total_loss', loss, epoch, iteration)
            optimizer.zero_grad()
            loss.backward()
            self._check_tensor('memory_ir.features_after_backward',
                               self.memory_ir.features, epoch, iteration)
            self._check_tensor('memory_rgb.features_after_backward',
                               self.memory_rgb.features, epoch, iteration)
            self._check_parameters('gradient_after_backward', epoch,
                                   iteration, gradients=True)
            optimizer.step()
            self._check_parameters('parameter_after_optimizer_step', epoch,
                                   iteration)
            losses.update(loss.item())
            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\tTime {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\tLoss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f}\tLoss rgb {:.3f}\t'.format(
                          epoch, i + 1, len(data_loader_rgb), batch_time.val,
                          batch_time.avg, data_time.val, data_time.avg,
                          losses.val, losses.avg, loss_ir, loss_rgb))
        self._print_agva_stats(epoch)


class ClusterContrastTrainer_PCLMP_NoEMALoss_AGVAMask(
        _AGVAMemoryMaskStats, ClusterContrastTrainer_PCLMP_NoEMALoss):
    def __init__(self, encoder, encoder_ema, memory=None,
                 mask_memory_updates=True):
        super(ClusterContrastTrainer_PCLMP_NoEMALoss_AGVAMask, self).__init__(
            encoder, encoder_ema, memory=memory)
        self._init_agva_stats(mask_memory_updates)

    @staticmethod
    def _parse_data_ir_flag(inputs):
        imgs, _, pids, _, indexes, is_aug = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda(), is_aug

    def train(self, epoch, data_loader_ir, data_loader_rgb,
              data_loader_all_ir, data_loader_all_rgb, optimizer,
              print_freq=10, train_iters=400, i2r=None, r2i=None):
        self.encoder.train()
        self.encoder_ema.train()
        self._init_agva_stats(self.mask_memory_updates)
        print('============================================')
        print('ABLATION: NO EMA LOSS ONLY + D-AGVA FLAGS')
        print('EMA loss included: False')
        print('ALL memory optimization enabled: True')
        print('D-AGVA memory updates masked: {}'.format(
            self.mask_memory_updates))
        print('============================================')
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()
        for i in range(train_iters):
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)
            inputs_ir, labels_ir, indexes_ir, is_aug = \
                self._parse_data_ir_flag(inputs_ir)
            update_mask_ir = self._record_aerial_batch(is_aug)
            inputs_rgb, inputs_rgb1, labels_rgb, indexes_rgb = \
                self._parse_data_rgb(inputs_rgb)
            inputs_rgb = torch.cat((inputs_rgb, inputs_rgb1), 0)
            labels_rgb = torch.cat((labels_rgb, labels_rgb), -1)
            (_, f_out_rgb, f_out_ir, labels_rgb, labels_ir,
             pool_rgb, pool_ir) = self._forward(
                 inputs_rgb, inputs_ir, label_1=labels_rgb,
                 label_2=labels_ir, modal=0)
            loss_ir = self._aerial_memory(
                self.memory_ir, f_out_ir, labels_ir, update_mask_ir)
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)

            if r2i:
                rgb2ir_labels = torch.tensor(
                    [r2i[key.item()] for key in labels_rgb]).cuda()
                ir2rgb_labels = torch.tensor(
                    [i2r[key.item()] for key in labels_ir]).cuda()
                if epoch % 2 == 1:
                    cross_loss = self._aerial_memory(
                        self.memory_rgb, f_out_ir, ir2rgb_labels.long(),
                        update_mask_ir)
                else:
                    cross_loss = self.memory_ir(
                        f_out_rgb, rgb2ir_labels.long())
            else:
                cross_loss = loss_ir.new_zeros(())

            with torch.no_grad():
                (_, f_out_rgb_ema, f_out_ir_ema, labels_rgb_ema,
                 labels_ir_ema, pool_rgb_ema, pool_ir_ema) = \
                    self._forward_ema(
                        inputs_rgb, inputs_ir, label_1=labels_rgb,
                        label_2=labels_ir, modal=0)
            loss_ir_ema = self._aerial_memory(
                self.memory_ir, f_out_ir_ema, labels_ir_ema,
                update_mask_ir, model_name='encoder_ema')
            loss_rgb_ema = self.memory_rgb(
                f_out_rgb_ema, labels_rgb_ema, model_name='encoder_ema')
            loss_ema = loss_ir_ema + loss_rgb_ema
            loss = compose_primary_loss_without_ema(
                loss_ir, loss_rgb, cross_loss)
            if i == 0:
                print('No EMA Loss ablation: loss_ema is monitored but '
                      'excluded from the optimized Stage 2 loss '
                      '(requires_grad={}).'.format(loss_ema.requires_grad))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.item())

            inputs_all_ir = data_loader_all_ir.next()
            inputs_all_rgb = data_loader_all_rgb.next()
            (inputs_all_ir, labels_all_ir, indexes_all_ir,
             is_aug_all) = self._parse_data_ir_flag(inputs_all_ir)
            update_mask_all_ir = self._record_aerial_batch(is_aug_all)
            (inputs_all_rgb, inputs_all_rgb1, labels_all_rgb,
             indexes_all_rgb) = self._parse_data_rgb(inputs_all_rgb)
            inputs_all_rgb = torch.cat(
                (inputs_all_rgb, inputs_all_rgb1), 0)
            labels_all_rgb = torch.cat(
                (labels_all_rgb, labels_all_rgb), -1)
            (_, f_out_all_rgb, f_out_all_ir, labels_all_rgb,
             labels_all_ir, pool_all_rgb, pool_all_ir) = self._forward(
                 inputs_all_rgb, inputs_all_ir, label_1=labels_all_rgb,
                 label_2=labels_all_ir, modal=0)
            loss_all_ir = self._aerial_memory(
                self.memory_all, f_out_all_ir, labels_all_ir,
                update_mask_all_ir)
            loss_all_rgb = self.memory_all(f_out_all_rgb, labels_all_rgb)
            loss2 = loss_all_ir + loss_all_rgb
            optimizer.zero_grad()
            loss2.backward()
            optimizer.step()
            self._update_ema_variables(self.encoder, self.encoder_ema, 0.999)

            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\tTime {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\tOptimized loss {:.3f} '
                      '({:.3f})\tLoss ir {:.3f}\tLoss rgb {:.3f}\t'
                      'EMA monitor (excluded) {:.3f}\tLoss ir ema {:.3f}\t'
                      'Loss rgb ema {:.3f}\tLoss all {:.3f}\t'
                      'Loss all ir {:.3f}\tLoss all rgb {:.3f}\t'.format(
                          epoch, i + 1, len(data_loader_rgb), batch_time.val,
                          batch_time.avg, data_time.val, data_time.avg,
                          losses.val, losses.avg, loss_ir, loss_rgb,
                          loss_ema, loss_ir_ema, loss_rgb_ema, loss2,
                          loss_all_ir, loss_all_rgb))
        self._print_agva_stats(epoch)
