def conv_output_size(in_size: int, kernel_size: int, stride: int, padding: int):
    return (in_size - kernel_size + 2 * padding) // stride + 1


def compute_conv_layer_sizes(in_size: int, conv_layers: list):

    current_size = in_size
    out_sizes = []
    for layer in conv_layers:
        current_size = conv_output_size(
            current_size, layer.kernel_size, layer.stride, layer.padding
        )
        out_sizes.append(current_size)

    return out_sizes
