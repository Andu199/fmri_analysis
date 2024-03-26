from torch import nn

ACTIVATIONS = {
    "relu": nn.ReLU(),
    "selu": nn.SELU(),
    "gelu": nn.GELU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "leaky_relu": nn.LeakyReLU(),
}

NORM_LAYERS = {
    "batchnorm1d": nn.BatchNorm1d,
    "batchnorm2d": nn.BatchNorm2d,
    "layernorm": nn.LayerNorm,
    "none": nn.Identity,
}


def get_linear_layer(config, input_dim, output_dim):
    layer = [nn.Linear(input_dim, output_dim)]

    activation = ACTIVATIONS.get(config["activation"], None)
    if activation is None:
        raise ValueError("Activation function not yet implemented: " + config["activation"])
    layer.append(activation)

    norm = NORM_LAYERS.get(config["norm_layer"], None)()
    if norm is None:
        raise ValueError("Norm layer not yet implemented: " + config["norm_layer"])
    layer.append(norm)

    if config["dropout_prob"] > 0.0:
        layer.append(nn.Dropout(p=config["dropout_prob"]))

    return nn.Sequential(*layer)
