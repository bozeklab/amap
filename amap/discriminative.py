from torch.nn.modules.loss import _Loss
import torch


def zero(gpu):
    res = torch.tensor(0.0)
    if gpu >= 0:
        res = res.cuda(gpu)
    return res

def calculate_means(pred, gt, n_objects, max_n_objects, gpu):
    """pred: bs, height * width, n_filters
       gt: bs, height * width, n_instances"""

    bs, n_loc, n_filters = pred.size()
    n_instances = gt.size(2)

    pred_repeated = pred.unsqueeze(2).expand(bs, n_loc, n_instances, n_filters)  # bs, n_loc, n_instances, n_filters
    # bs, n_loc, n_instances, 1
    gt_expanded = gt.unsqueeze(3)

    pred_masked = pred_repeated * gt_expanded

    means = []
    for i in range(bs):
        _n_objects_sample = n_objects[i]
        # n_loc, n_objects, n_filters
        _pred_masked_sample = pred_masked[i, :, : _n_objects_sample]
        # n_loc, n_objects, 1
        _gt_expanded_sample = gt_expanded[i, :, : _n_objects_sample]

        _mean_sample = _pred_masked_sample.sum(0) / _gt_expanded_sample.sum(0)  # n_objects, n_filters
        if (max_n_objects - _n_objects_sample) != 0:
            n_fill_objects = int(max_n_objects - _n_objects_sample)
            _fill_sample = torch.zeros(n_fill_objects, n_filters)
            if gpu >= 0:
                _fill_sample = _fill_sample.cuda(gpu)
            # Variable(_fill_sample)
            _fill_sample.requires_grad_()
            _mean_sample = torch.cat((_mean_sample, _fill_sample), dim=0)
        means.append(_mean_sample)

    means = torch.stack(means)

    # means = pred_masked.sum(1) / gt_expanded.sum(1)
    # # bs, n_instances, n_filters

    return means


def calculate_variance_term(pred, gt, means, n_objects, delta_v, norm=2, gpu=-1):
    """pred: bs, height * width, n_filters
       gt: bs, height * width, n_instances
       means: bs, n_instances, n_filters"""

    bs, n_loc, n_filters = pred.size()
    n_instances = gt.size(2)

    # bs, n_loc, n_instances, n_filters
    means = means.unsqueeze(1).expand(bs, n_loc, n_instances, n_filters)
    # bs, n_loc, n_instances, n_filters
    pred = pred.unsqueeze(2).expand(bs, n_loc, n_instances, n_filters)
    # bs, n_loc, n_instances, n_filters
    gt = gt.unsqueeze(3).expand(bs, n_loc, n_instances, n_filters)

    _var = (torch.clamp(torch.norm((pred - means), norm, 3) - delta_v, min=0.0) ** 2) * gt[:, :, :, 0]

    var_term = zero(gpu) # 0.0
    bs_ = 0
    for i in range(bs):
        if n_objects[i] > 0:
            _var_sample = _var[i, :, :n_objects[i]]  # n_loc, n_objects
            _gt_sample = gt[i, :, :n_objects[i], 0]  # n_loc, n_objects

            var_term += torch.sum(_var_sample) / torch.sum(_gt_sample)
            bs_ += 1
    if bs_ > 0:
        var_term = var_term / bs_

    return var_term


def calculate_distance_term(means, n_objects, delta_d, norm=2, gpu=-1):
    """means: bs, n_instances, n_filters"""

    bs, n_instances, n_filters = means.size()

    dist_term = zero(gpu) #0.0
    bs_ = 0
    for i in range(bs):
        _n_objects_sample = int(n_objects[i])

        if _n_objects_sample <= 1:
            continue

        bs_ += 1
        _mean_sample = means[i, : _n_objects_sample, :]  # n_objects, n_filters
        means_1 = _mean_sample.unsqueeze(1).expand(_n_objects_sample, _n_objects_sample, n_filters)
        means_2 = means_1.permute(1, 0, 2)

        diff = means_1 - means_2  # n_objects, n_objects, n_filters

        _norm = torch.norm(diff, norm, 2)

        margin = 2 * delta_d * (1.0 - torch.eye(_n_objects_sample))
        if gpu >=0:
            margin = margin.cuda(gpu)
        #margin = Valiable(margin)
        margin.requires_grad_()

        _dist_term_sample = torch.sum(torch.clamp(margin - _norm, min=0.0) ** 2)
        _dist_term_sample = _dist_term_sample / (_n_objects_sample * (_n_objects_sample - 1))
        dist_term += _dist_term_sample

    if bs_ > 0:
        dist_term = dist_term / bs_

    return dist_term


def calculate_regularization_term(means, n_objects, norm, gpu=-1):
    """means: bs, n_instances, n_filters"""

    bs, n_instances, n_filters = means.size()

    reg_term = zero(gpu) #0.0
    bs_ = 0
    for i in range(bs):
        if n_objects[i] > 0:
            _mean_sample = means[i, : n_objects[i], :]  # n_objects, n_filters
            _norm = torch.norm(_mean_sample, norm, 1)
            reg_term += torch.mean(_norm)
            bs_ += 1
    if bs_ > 0:
        reg_term = reg_term / bs_

    return reg_term


def discriminative_loss(input, target, n_objects, delta_v, delta_d, norm, gpu):
    """input: bs, n_filters, fmap, fmap
       target: bs, n_instances, fmap, fmap
       n_objects: bs"""

    alpha = beta = 1.0
    gamma = 0.001

    bs, n_filters, height, width = input.size()
    max_n_objects = target.size(1)

    input = input.permute(0, 2, 3, 1).contiguous().view(bs, height * width, n_filters)
    target = target.permute(0, 2, 3, 1).contiguous().view(bs, height * width, max_n_objects)

    cluster_means = calculate_means(input, target, n_objects, max_n_objects, gpu)

    var_term = calculate_variance_term(input, target, cluster_means, n_objects, delta_v, norm, gpu)
    dist_term = calculate_distance_term(cluster_means, n_objects, delta_d, norm, gpu)
    reg_term = calculate_regularization_term(cluster_means, n_objects, norm, gpu)

    loss = alpha * var_term + beta * dist_term + gamma * reg_term

    return loss


class DiscriminativeLoss(_Loss):

    def __init__(self, delta_var, delta_dist, norm, size_average=True, reduce=True, gpu=-1):
        super(DiscriminativeLoss, self).__init__(size_average)
        self.reduce = reduce

        assert size_average
        assert self.reduce

        self.delta_var = float(delta_var)
        self.delta_dist = float(delta_dist)
        self.norm = int(norm)
        self.gpu = gpu

        assert self.norm in [1, 2]

    def forward(self, input, target, n_objects):
        #_assert_no_grad(target)
        return discriminative_loss(input, target, n_objects, self.delta_var, self.delta_dist, self.norm, self.gpu)