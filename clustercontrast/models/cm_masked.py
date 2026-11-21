"""Cluster memories whose backward update can ignore selected samples.

The mask changes only the in-place memory-bank update.  Forward logits and
input gradients still use every sample, so augmented images keep contributing
to the optimization loss.
"""

from __future__ import absolute_import

import collections

import numpy as np
import torch
import torch.nn.functional as F
from torch import autograd

from .cm import ClusterMemory, cm, cm_hard, cm_hybrid


class CMMasked(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, targets, features, momentum, update_mask):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets, update_mask)
        return inputs.mm(features.t())

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets, update_mask = ctx.saved_tensors
        grad_inputs = (grad_outputs.mm(ctx.features)
                       if ctx.needs_input_grad[0] else None)
        for x, y, update in zip(inputs, targets, update_mask):
            if not bool(update):
                continue
            ctx.features[y] = (
                ctx.momentum * ctx.features[y]
                + (1.0 - ctx.momentum) * x)
            ctx.features[y] /= ctx.features[y].norm()
        return grad_inputs, None, None, None, None


class CMHybridMasked(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, targets, features, momentum, update_mask):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets, update_mask)
        return inputs.mm(features.t())

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets, update_mask = ctx.saved_tensors
        nums = len(ctx.features) // 2
        grad_inputs = (grad_outputs.mm(ctx.features)
                       if ctx.needs_input_grad[0] else None)

        batch_centers = collections.defaultdict(list)
        for feature, index, update in zip(
                inputs, targets.tolist(), update_mask.tolist()):
            if update:
                batch_centers[index].append(feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(
                    ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())
            mean = torch.stack(features, dim=0).mean(0)
            ctx.features[index] = (
                ctx.features[index] * ctx.momentum
                + (1.0 - ctx.momentum) * mean)
            ctx.features[index] /= ctx.features[index].norm()
            hard = np.argmin(np.array(distances))
            ctx.features[index + nums] = (
                ctx.features[index + nums] * ctx.momentum
                + (1.0 - ctx.momentum) * features[hard])
            ctx.features[index + nums] /= ctx.features[index + nums].norm()
        return grad_inputs, None, None, None, None


class CMHardMasked(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, targets, features, momentum, num_instances,
                update_mask):
        ctx.features = features
        ctx.num_instances = num_instances
        ctx.save_for_backward(inputs, targets, update_mask)
        return inputs.mm(features.t())

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets, update_mask = ctx.saved_tensors
        nums = len(ctx.features) // ctx.num_instances
        grad_inputs = (grad_outputs.mm(ctx.features)
                       if ctx.needs_input_grad[0] else None)

        batch_centers = collections.defaultdict(list)
        for feature, index, update in zip(
                inputs, targets.tolist(), update_mask.tolist()):
            if update:
                batch_centers[index].append(feature)
        for index, features in batch_centers.items():
            indexes = [index + nums * i
                       for i in range(ctx.num_instances)]
            replace = len(features) < 16
            ids = np.random.choice(
                len(features), len(indexes), replace=replace)
            ctx.features[indexes] = torch.stack(features)[ids]
        return grad_inputs, None, None, None, None, None


def cm_masked(inputs, indexes, features, update_mask, momentum=0.5):
    return CMMasked.apply(
        inputs, indexes, features,
        torch.Tensor([momentum]).to(inputs.device), update_mask)


def cm_hybrid_masked(inputs, indexes, features, update_mask, momentum=0.5):
    return CMHybridMasked.apply(
        inputs, indexes, features,
        torch.Tensor([momentum]).to(inputs.device), update_mask)


def cm_hard_masked(inputs, indexes, features, update_mask, momentum=0.5,
                   num_instances=16):
    return CMHardMasked.apply(
        inputs, indexes, features,
        torch.Tensor([momentum]).to(inputs.device), num_instances,
        update_mask)


class MaskedClusterMemory(ClusterMemory):
    """Drop-in ``ClusterMemory`` with an update-only sample mask."""

    def forward(self, inputs, targets, model_name='encoder',
                update_mask=None):
        # Calling the original implementation is important here: it preserves
        # both bit-level arithmetic and NumPy RNG consumption for old runs.
        if update_mask is None:
            return super(MaskedClusterMemory, self).forward(
                inputs, targets, model_name=model_name)
        update_mask = torch.as_tensor(
            update_mask, dtype=torch.bool, device=targets.device).reshape(-1)
        if update_mask.numel() != targets.numel():
            raise ValueError(
                "update_mask has {} values for {} targets".format(
                    update_mask.numel(), targets.numel()))
        if bool(update_mask.all()):
            return super(MaskedClusterMemory, self).forward(
                inputs, targets, model_name=model_name)

        inputs = F.normalize(inputs, dim=1).cuda()
        if self.cm_type == 'CM':
            outputs = cm_masked(
                inputs, targets, self.features, update_mask, self.momentum)
            outputs /= self.temp
            return self.cross_entropy(outputs, targets)

        if self.cm_type == 'CMhybrid':
            outputs = cm_hybrid_masked(
                inputs, targets, self.features, update_mask, self.momentum)
            outputs /= self.temp
            mean, hard = torch.chunk(outputs, 2, dim=1)
            return 0.5 * (
                self.cross_entropy(hard, targets)
                + torch.relu(self.cross_entropy(mean, targets) - 0.2))

        if self.cm_type == 'CMhard' and model_name == 'encoder':
            outputs = cm_hybrid_masked(
                inputs, targets, self.features, update_mask, self.momentum)
            outputs /= self.temp
            mean, hard = torch.chunk(outputs, 2, dim=1)
            return 0.5 * (
                self.cross_entropy(hard, targets)
                + torch.relu(self.cross_entropy(mean, targets) - 0.2))

        if self.cm_type == 'CMhard' and model_name == 'encoder_ema':
            outputs = cm_hard_masked(
                inputs, targets, self.features_ema, update_mask,
                self.momentum, self.num_instances)
            outputs /= self.temp
            out = torch.stack(torch.chunk(
                outputs, self.num_instances, dim=1), dim=0)
            neg = torch.max(out, dim=0)[0]
            pos = torch.min(out, dim=0)[0]
            mask = torch.zeros_like(out[0]).scatter_(
                1, targets.unsqueeze(1), 1)
            mask_in = mask * pos + (1 - mask) * neg
            return self.cross_entropy(mask_in, targets)

        raise ValueError(
            "unsupported memory/model combination: {}/{}".format(
                self.cm_type, model_name))
