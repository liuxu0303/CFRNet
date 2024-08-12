import torch.nn as nn
import torch.nn.functional as f
from .lit_transformer import LiteMono
from .submodules import \
    ConvLayer, UpsampleConvLayer, TransposedConvLayer, \
    CFMF, TCM

class BaseNet(nn.Module):
    def __init__(self, num_bins, base_num_channels, num_output_channels,
                 use_upsample_conv, norm, kernel_size=5):
        super(BaseNet, self).__init__()
        self.base_num_channels = base_num_channels
        self.num_output_channels = num_output_channels
        self.kernel_size = kernel_size
        self.norm = norm
        self.num_bins = num_bins
        self.encoder_input_sizes = [32, 32, 64]
        self.encoder_output_sizes = [32, 64, 128]
        self.max_num_channels = self.encoder_output_sizes[-1]
        if use_upsample_conv:
            print('Using UpsampleConvLayer (slow, but no checkerboard artefacts)')
            self.UpsampleLayer = UpsampleConvLayer
        else:
            print('Using TransposedConvLayer (fast, with checkerboard artefacts)')
            self.UpsampleLayer = TransposedConvLayer
        assert(self.num_output_channels > 0)
        print(f'Kernel size {self.kernel_size}')
        print(f'norm {self.norm}')

    def build_decoders(self):
        decoder_input_sizes = reversed(self.encoder_output_sizes)
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        decoders = nn.ModuleList()
        for input_size, output_size in zip(decoder_input_sizes, decoder_output_sizes):
            decoders.append(self.UpsampleLayer(
                input_size if self.skip_type == 'sum' else 2 * input_size,
                output_size, kernel_size=self.kernel_size,
                padding=self.kernel_size // 2, norm=self.norm))
        return decoders

    def build_prediction_layer(self, num_output_channels, norm=None):
        return ConvLayer(self.base_num_channels if self.skip_type == 'sum' else 2 * self.base_num_channels,
                         num_output_channels, 1, activation=None, norm=norm)


class EventLiteMono(BaseNet):
    """
    Baseline1 with single event modality
    """

    def __init__(self, net_kwargs):
        super().__init__(**net_kwargs)
        self.final_activation = nn.Sigmoid()
        # Event encoder
        self.event_encoder = LiteMono(self.num_bins, model='lite-mono')

        self.decoders = self.build_decoders()
        self.pred = ConvLayer(self.base_num_channels, self.num_output_channels, kernel_size=1, activation=None)

    def forward(self, x):

        # encoder
        x = self.event_encoder(x)[-1]

        # decoder
        for i, decoder in enumerate(self.decoders):
            x = decoder(x)

        x = f.interpolate(x, scale_factor=2, mode='bicubic', align_corners=True)
        depth = self.final_activation(self.pred(x))

        return {'pred_depth': depth}

class RGBLiteMono(BaseNet):
    """
    Baseline2 with single RGB frame modality
    """

    def __init__(self, net_kwargs):
        super().__init__(**net_kwargs)
        self.final_activation = nn.Sigmoid()

        self.RGB_encoder = LiteMono(3, model='lite-mono')

        self.decoders = self.build_decoders()
        self.pred = ConvLayer(self.base_num_channels, self.num_output_channels, kernel_size=1, activation=None)

    def forward(self, y):

        # encoder
        y = self.RGB_encoder(y)[-1]

        # decoder
        for i, decoder in enumerate(self.decoders):
            y = decoder(y)

        y = f.interpolate(y, scale_factor=2, mode='bicubic', align_corners=True)
        depth = self.final_activation(self.pred(y))

        return {'pred_depth': depth}


class CFRNet(BaseNet):
    """
    the proposed high-rate monocular depth estimator
    using a cross frame-rate frame-event joint learning network (CFRNet).
    """
    def __init__(self, net_kwargs):
        super().__init__(**net_kwargs)
        self.final_activation = nn.Sigmoid()

        # the modality-specific shared encoders adopting a
        # lightweight CNN-Transformer hybrid backbone
        # Event encoder
        self.event_encoder = LiteMono(self.num_bins, model='lite-mono')
        # RGB frame encoder
        self.RGB_encoder = LiteMono(3, model='lite-mono')

        # Cross Frame-rate Multi-modal Fusion (CFMF) utilizes implicit spatial
        # alignment and dynamic attention-based fusion strategies
        self.cfmf = CFMF(self.max_num_channels)

        # Temporal Consistent Modeling (TCM) adopting
        # the recurrent structure
        self.state = None
        self.tcm = TCM(input_size=self.max_num_channels, hidden_size=self.max_num_channels)

        # normal CNN-based decoder
        self.decoders = self.build_decoders()
        self.pred = ConvLayer(self.base_num_channels, self.num_output_channels, kernel_size=1, activation=None)

    def forward(self, x, y):
        """
        :param x: N x num_input_channels x H x W
        :param y: N x num_input_channels x H x W
        :return: N x num_output_channels x H x W
        """

        # extract local-global features
        # from two heterogeneous streams
        x = self.event_encoder(x)[-1]
        y = self.RGB_encoder(y)[-1]

        # generate a complementary
        # joint representation
        fu = self.cfmf(x, y)

        # model long-range temporal dependencies
        # between the joint representations
        st, self.state = self.tcm(fu, self.state)

        # predict high-rate and fine-grained depth maps
        for i, decoder in enumerate(self.decoders):
            st = decoder(st)
        st = f.interpolate(st, scale_factor=2, mode='bicubic', align_corners=True)
        depth = self.final_activation(self.pred(st))

        return {'pred_depth': depth}
