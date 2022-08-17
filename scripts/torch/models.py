""" convolution neural network for structure guided deblurring and denoising """
import abc
from collections import OrderedDict
from typing import Optional

import torch


class BlockGenerator(abc.ABC):
    """ abstract base class for convolutional block with batch norm and activation """

    @abc.abstractmethod
    def __call__(self, in_channels: int, out_channels: int, groups: int):
        raise NotImplementedError


class SimpleBlockGenerator(BlockGenerator):
    """ simple convolutional block with batch norm and activation """

    def __init__(self,
                 kernel_size: tuple[int, int, int] = (3, 3, 3),
                 activation: Optional[torch.nn.Module] = None,
                 batchnorm: bool = True,
                 padding: str = 'same'):
        self.kernel_size = kernel_size
        self.activation = activation
        self.batchnorm = batchnorm
        self.padding = 'same'

    def __call__(self, in_channels, out_channels, groups):
        od = OrderedDict()

        od['conv'] = torch.nn.Conv3d(in_channels=in_channels,
                                     out_channels=out_channels,
                                     groups=groups,
                                     kernel_size=self.kernel_size,
                                     padding=self.padding)

        if self.batchnorm:
            od['bnorm'] = torch.nn.BatchNorm3d(out_channels)

        if self.activation is None:
            od['act'] = torch.nn.PReLU(num_parameters=out_channels)
        else:
            od['act'] = self.activation

        return torch.nn.Sequential(od)


class SequentialStructureConvNet(torch.nn.Module):
    """ conv net for structure guided deconvolution / denoising """

    def __init__(self,
                 num_input_ch: int,
                 block_generator: BlockGenerator,
                 nblocks: int = 8,
                 nfeat: int = 30):
        super(SequentialStructureConvNet, self).__init__()

        self.num_input_ch = num_input_ch
        self.nfeat = nfeat
        self.nblocks = nblocks

        od = OrderedDict()
        # add first block
        od['b0'] = block_generator(self.num_input_ch, self.nfeat,
                                   self.num_input_ch)

        # middle blocks
        for i in range(self.nblocks):
            od[f'b{i+1}'] = block_generator(self.nfeat, self.nfeat, 1)

        od['conv111'] = torch.nn.Conv3d(in_channels=self.nfeat,
                                        out_channels=1,
                                        kernel_size=(1, 1, 1),
                                        padding='same')

        self.layer_stack = torch.nn.Sequential(od)

    def forward(self, x: torch.Tensor):
        """ forward path including residual and ReLU """
        return torch.nn.ReLU()(self.layer_stack(x) + x[:, 0:1, ...])


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    x = torch.rand(10, 2, 29, 20, 29, device=device)

    bgen = SimpleBlockGenerator()
    model = APetNet(x.shape[1], bgen, nfeat=4, nblocks=3).to(device)

    pred = model.forward(x)
    print(model)