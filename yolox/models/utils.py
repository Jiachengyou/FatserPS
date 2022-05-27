import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
class NormAwareEmbedding(nn.Module):
    """
    Implements the Norm-Aware Embedding proposed in
    Chen, Di, et al. "Norm-aware embedding for efficient person search." CVPR 2020.
    """

    def __init__(self, in_channels=1024, dim=256):
        super(NormAwareEmbedding, self).__init__()
        self.in_channels = in_channels
        self.dim = dim

        proj = nn.Sequential(nn.Linear(in_channels, dim), nn.BatchNorm1d(dim))
        

        init.normal_(proj[0].weight, std=0.01)
        init.constant_(proj[0].bias, 0)
        
        init.normal_(proj[1].weight, std=0.01)
        init.constant_(proj[1].bias, 0)
        
        self.projectors = proj

        self.rescaler = nn.BatchNorm1d(1, affine=True)

    def forward(self, x):
        """
        Arguments:
            featmaps: OrderedDict[Tensor], and in featmap_names you can choose which
                      featmaps to use
        Returns:
            tensor of size (BatchSize, dim), L2 normalized embeddings.
            tensor of size (BatchSize, ) rescaled norm of embeddings, as class_logits.
        """
        B,D,H,W = x.shape
        x = x.permute(0,2,3,1)
        x = x.contiguous().view(B*H*W, D)
        
        embeddings = self.projectors(x)
        norms = embeddings.norm(2, -1, keepdim=True)
        embeddings = embeddings / norms.expand_as(embeddings).clamp(min=1e-12)
        norms = self.rescaler(norms)
        embeddings = embeddings.contiguous().view(B,H,W,D).permute(0,3,1,2)
        norms = norms.contiguous().view(B,H,W,1).permute(0,3,1,2)
        return embeddings, norms

    def _flatten_fc_input(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            return x.flatten(start_dim=1)
        return x
    
    
class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.
    This layer calculates the target location by :math: `sum{P(y_i) * y_i}`,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}
    Args:
        reg_max (int): The maximal value of the discrete set. Default: 16. You
            may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.
        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.
        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 4)
        return x
    
def knowledge_distillation_kl_div_loss(pred, soft_label, T=10, detach_target=True):
    r"""Loss function for knowledge distilling using KL divergence.
    Args:
        pred (Tensor): Predicted logits with shape (N, n + 1).
        soft_label (Tensor): Target logits with shape (N, N + 1).
        T (int): Temperature for distillation.
        detach_target (bool): Remove soft_label from automatic differentiation
    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    assert pred.size() == soft_label.size()
    target = F.softmax(soft_label / T, dim=1)
    if detach_target:
        target = target.detach()

    kd_loss = F.kl_div(
        F.log_softmax(pred / T, dim=1), target, reduction='none').mean(1) * (
            T * T)

    return kd_loss
