import math
import warnings
import torch.nn as nn
# local modules
from .base.base_model import BaseModel

from .net import EventLiteMono, RGBLiteMono, CFRNet


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class Event2Depth(BaseModel):
    """
    Baseline1 with single event modality
    """

    def __init__(self, net_kwargs):
        super().__init__(net_kwargs)

        self.net = EventLiteMono(net_kwargs)

    def reset_states(self):
        self.net.state = None

    def init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, event_tensor):
        """
        :param event_tensor: N x num_bins x H x W
        :return: output dict with mask taking values in [0,1], and
                 depth prediction
        """
        output_dict = self.net(event_tensor)

        return output_dict

class RGB2Depth(BaseModel):
    """
    Baseline2 with single RGB frame modality
    """

    def __init__(self, net_kwargs):
        super().__init__(net_kwargs)

        self.net = RGBLiteMono(net_kwargs)

    def init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, RGB_tensor):
        """
        :param rgb_tensor: N x 3 x H x W
        :return: output dict with mask taking values in [0,1], and
                 depth prediction
        """
        output_dict = self.net(RGB_tensor)

        return output_dict

class CFRNet2Depth(BaseModel):
    """
    the proposed high-rate monocular depth estimator using a cross frame-rate frame-event joint learning network (CFRNet).
    """

    def __init__(self, net_kwargs):
        super().__init__(net_kwargs)

        self.net = CFRNet(net_kwargs)

    def reset_states(self):
        self.net.state = None

    def init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, event_tensor, RGB_tensor):

        output_dict = self.net(event_tensor, RGB_tensor)

        return output_dict