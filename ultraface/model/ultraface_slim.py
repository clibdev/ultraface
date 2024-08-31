from torch.nn import Conv2d, Sequential, ModuleList, ReLU
from ultraface.model.mb_tiny import Mb_Tiny
from ultraface.model.ssd import SSD
from ultraface.utils import box_utils


class UltraFaceSlim:
    num_classes = 2
    source_layer_indexes = [8, 11, 13]
    center_variance = 0.1
    size_variance = 0.2

    def __init__(self, size, device):
        self.device = device
        base_net = Mb_Tiny(self.num_classes)
        base_net_model = base_net.model

        extras = ModuleList([
            Sequential(
                Conv2d(base_net.base_channel * 16, base_net.base_channel * 4, 1),
                ReLU(),
                self.seperable_conv_2d(base_net.base_channel * 4, base_net.base_channel * 16, 3, 2, 1),
                ReLU()
            )
        ])
        regression_headers = ModuleList([
            self.seperable_conv_2d(base_net.base_channel * 4, 3 * 4, 3, 1),
            self.seperable_conv_2d(base_net.base_channel * 8, 2 * 4, 3, 1),
            self.seperable_conv_2d(base_net.base_channel * 16, 2 * 4, 3, 1),
            Conv2d(base_net.base_channel * 16, 3 * 4, 3, 1)
        ])
        classification_headers = ModuleList([
            self.seperable_conv_2d(base_net.base_channel * 4, 3 * self.num_classes, 3, 1),
            self.seperable_conv_2d(base_net.base_channel * 8, 2 * self.num_classes, 3, 1),
            self.seperable_conv_2d(base_net.base_channel * 16, 2 * self.num_classes, 3, 1),
            Conv2d(base_net.base_channel * 16, 3 * self.num_classes, 3, padding=1)
        ])
        self.net = SSD(
            self.num_classes,
            base_net_model,
            self.source_layer_indexes,
            extras,
            classification_headers,
            regression_headers,
            self.center_variance,
            self.size_variance,
            box_utils.generate_priors(size),
            is_test=True,
            device=device
        )

    @staticmethod
    def seperable_conv_2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        return Sequential(
            Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels),
            ReLU(),
            Conv2d(in_channels, out_channels, 1),
        )
