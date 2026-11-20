# -*- coding: utf-8 -*-
"""No-EMA-loss Stage-2 trainer with partial cross-view correspondences."""

from __future__ import absolute_import, print_function

import time

import torch

from .trainers_no_ema_loss import (
    ClusterContrastTrainer_PCLMP_NoEMALoss,
    compose_primary_loss_without_ema)
from .utils.meters import AverageMeter


def mapped_memory_loss(memory, features, source_labels, mapping):
    """Average memory loss over samples whose source label has a mapping."""
    mapping = {} if mapping is None else mapping
    labels = [int(value) for value in source_labels.detach().cpu().tolist()]
    valid_indexes = [
        index for index, label in enumerate(labels) if label in mapping]
    if not valid_indexes:
        return features.new_zeros(())
    index_tensor = torch.as_tensor(
        valid_indexes, dtype=torch.long, device=features.device)
    selected_features = features.index_select(0, index_tensor)
    mapped_targets = torch.as_tensor(
        [int(mapping[labels[index]]) for index in valid_indexes],
        dtype=torch.long, device=features.device)
    return memory(selected_features, mapped_targets)


def confident_cross_view_loss(epoch, memory_ir, memory_rgb,
                              features_rgb, features_ir,
                              labels_rgb, labels_ir, i2r, r2i):
    """Keep the baseline alternating direction while allowing partial maps."""
    if int(epoch) % 2 == 1:
        return mapped_memory_loss(
            memory_rgb, features_ir, labels_ir, i2r)
    return mapped_memory_loss(
        memory_ir, features_rgb, labels_rgb, r2i)


class ClusterContrastTrainer_PCLMP_ConfidentCross(
        ClusterContrastTrainer_PCLMP_NoEMALoss):
    """Change only the baseline cross-view term to accept partial mappings."""

    def train(self, epoch, data_loader_ir, data_loader_rgb, data_loader_all_ir,
              data_loader_all_rgb, optimizer, print_freq=10, train_iters=400,
              i2r=None, r2i=None):
        self.encoder.train()
        self.encoder_ema.train()

        print("============================================")
        print("ABLATION: CONFIDENCE-AWARE CROSS-VIEW MAPPING")
        print("Trainer source: {}".format(__file__))
        print("EMA loss included: False")
        print("Cross-view loss uses mapped samples only: True")
        print("ALL memory optimization enabled: True")
        print("============================================")
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)

            inputs_ir, labels_ir, indexes_ir = self._parse_data_ir(inputs_ir)
            inputs_rgb, inputs_rgb1, labels_rgb, indexes_rgb = \
                self._parse_data_rgb(inputs_rgb)

            inputs_rgb = torch.cat((inputs_rgb, inputs_rgb1), 0)
            labels_rgb = torch.cat((labels_rgb, labels_rgb), -1)
            _, f_out_rgb, f_out_ir, labels_rgb, labels_ir, pool_rgb, pool_ir = \
                self._forward(inputs_rgb, inputs_ir, label_1=labels_rgb,
                              label_2=labels_ir, modal=0)

            loss_ir = self.memory_ir(f_out_ir, labels_ir)
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)
            cross_loss = confident_cross_view_loss(
                epoch, self.memory_ir, self.memory_rgb,
                f_out_rgb, f_out_ir, labels_rgb, labels_ir,
                i2r, r2i)

            # Preserve the original EMA monitoring path; only loss_ema stays
            # excluded, exactly as in ClusterContrastTrainer_PCLMP_NoEMALoss.
            with torch.no_grad():
                (_, f_out_rgb_ema, f_out_ir_ema, labels_rgb_ema,
                 labels_ir_ema, pool_rgb_ema, pool_ir_ema) = self._forward_ema(
                    inputs_rgb, inputs_ir, label_1=labels_rgb,
                    label_2=labels_ir, modal=0)
            loss_ir_ema = self.memory_ir(
                f_out_ir_ema, labels_ir_ema, model_name="encoder_ema")
            loss_rgb_ema = self.memory_rgb(
                f_out_rgb_ema, labels_rgb_ema, model_name="encoder_ema")
            loss_ema = loss_ir_ema + loss_rgb_ema

            loss = compose_primary_loss_without_ema(
                loss_ir, loss_rgb, cross_loss)

            if i == 0:
                print("Confidence-aware cross-view loss: weight=0.25, "
                      "direction={}".format(
                          "i2r" if epoch % 2 == 1 else "r2i"))
                print("No EMA Loss ablation: loss_ema is monitored but "
                      "excluded from the optimized Stage 2 loss "
                      "(requires_grad={}).".format(loss_ema.requires_grad))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.item())

            inputs_all_ir = data_loader_all_ir.next()
            inputs_all_rgb = data_loader_all_rgb.next()
            inputs_all_ir, labels_all_ir, indexes_all_ir = \
                self._parse_data_ir(inputs_all_ir)
            inputs_all_rgb, inputs_all_rgb1, labels_all_rgb, indexes_all_rgb = \
                self._parse_data_rgb(inputs_all_rgb)

            inputs_all_rgb = torch.cat((inputs_all_rgb, inputs_all_rgb1), 0)
            labels_all_rgb = torch.cat((labels_all_rgb, labels_all_rgb), -1)
            (_, f_out_all_rgb, f_out_all_ir, labels_all_rgb, labels_all_ir,
             pool_all_rgb, pool_all_ir) = self._forward(
                inputs_all_rgb, inputs_all_ir, label_1=labels_all_rgb,
                label_2=labels_all_ir, modal=0)

            loss_all_ir = self.memory_all(f_out_all_ir, labels_all_ir)
            loss_all_rgb = self.memory_all(f_out_all_rgb, labels_all_rgb)
            loss2 = loss_all_ir + loss_all_rgb

            optimizer.zero_grad()
            loss2.backward()
            optimizer.step()
            self._update_ema_variables(self.encoder, self.encoder_ema, 0.999)

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Optimized loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f}\t'
                      'Loss rgb {:.3f}\t'
                      'Cross loss {:.3f}\t'
                      'EMA monitor (excluded) {:.3f}\t'
                      'Loss ir ema {:.3f}\t'
                      'Loss rgb ema {:.3f}\t'
                      'Loss all {:.3f}\t'
                      'Loss all ir {:.3f}\t'
                      'Loss all rgb {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg, loss_ir, loss_rgb,
                              cross_loss, loss_ema, loss_ir_ema,
                              loss_rgb_ema, loss2, loss_all_ir, loss_all_rgb))
