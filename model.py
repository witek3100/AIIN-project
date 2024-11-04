import torch.nn as nn


class Model(nn.Module):
    def __init__(self, **kwargs):
        super(Model, self).__init__()

        num_conv_layers = kwargs.get('num_conv_layers')
        conv_kernels_sizes = kwargs.get('conv_kernels_sizes')
        pool_kernels_sizes = kwargs.get('pool_kernels_sizes')
        conv_neurons = kwargs.get('conv_neurons')
        paddings = kwargs.get('paddings')

        conv_layers = []
        last_layer = 0
        for i in range(num_conv_layers):
            in_channels = conv_neurons[i-1] if i > 0 else 3
            conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, conv_neurons[i], kernel_size=conv_kernels_sizes[i], padding=paddings[i]),
                    nn.BatchNorm2d(conv_neurons[i]),
                    nn.ReLU(),
                )
            )
            last_layer = i
        self.conv_layers = nn.Sequential(*conv_layers)

        self.linear_layer = nn.Linear(conv_neurons[last_layer-1] * 8 * 8, 10)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.linear_layer(x)

        return x