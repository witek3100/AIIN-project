import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, **kwargs):
        super(Model, self).__init__()

        num_conv_layers = kwargs.get('num_conv_layers', 3)
        conv_kernels_sizes = kwargs.get('conv_kernels_sizes', [3, 3, 3])
        pool_kernels_sizes = kwargs.get('pool_kernels_sizes', [2, 2, 2])
        conv_neurons = kwargs.get('conv_neurons', [16, 32, 64])
        paddings = kwargs.get('paddings', [1, 1, 1])

        assert len(conv_kernels_sizes) == num_conv_layers, "Length of conv_kernels_sizes must match num_conv_layers"
        assert len(pool_kernels_sizes) == num_conv_layers, "Length of pool_kernels_sizes must match num_conv_layers"
        assert len(conv_neurons) == num_conv_layers, "Length of conv_neurons must match num_conv_layers"
        assert len(paddings) == num_conv_layers, "Length of paddings must match num_conv_layers"

        self.features = nn.ModuleList()
        current_channels = input_channels

        for i in range(num_conv_layers):
            conv_layer = nn.Conv2d(
                in_channels=current_channels,
                out_channels=conv_neurons[i],
                kernel_size=conv_kernels_sizes[i],
                padding=paddings[i]
            )

            self.features.extend([
                conv_layer,
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=pool_kernels_sizes[i])
            ])

            current_channels = conv_neurons[i]

        flatten_size = self._get_flatten_size(input_channels)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def _get_flatten_size(self, input_channels, input_size=(12, 12)):
        """
        Dynamically calculate the size of the flattened layer
        by creating a dummy input and passing it through the feature layers.
        """
        test_input = torch.zeros(1, input_channels, *input_size)
        feature_sizes = []

        with torch.no_grad():
            x = test_input

            for layer in self.features:
                x = layer(x)
                if isinstance(layer, nn.MaxPool2d):
                    feature_sizes.append(x.size())

        last_feature_size = feature_sizes[-1]
        flatten_size = last_feature_size[1] * last_feature_size[2] * last_feature_size[3]
        return flatten_size

    def forward(self, x):
        """
        Forward pass through the network.
        """
        for layer in self.features:
            x = layer(x)

        return self.classifier(x)


def create_model(input_channels, num_classes, input_size=(12, 12), **kwargs):
    """
    Helper function to create the dynamic CNN model.

    Args:
    - input_channels: Number of input channels
    - num_classes: Number of output classes
    - input_size: Size of input image (height, width)
    - **kwargs: Additional model configuration parameters
    """
    return Model(input_channels, num_classes, **kwargs)
