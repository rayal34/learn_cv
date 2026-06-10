from unittest.mock import patch

import numpy as np
import torch
from mnist.viz_utils import show_image


@patch("mnist.viz_utils.plt")
def test_show_image(mock_plt):
    # Create a dummy image tensor of shape (3, 28, 28)
    dummy_img = torch.randn(3, 28, 28)

    show_image(dummy_img)

    # Verify plt.imshow is called
    mock_plt.imshow.assert_called_once()
    # Verify plt.show is called
    mock_plt.show.assert_called_once()

    # Verify the argument to imshow has transposed dimensions (28, 28, 3)
    call_args = mock_plt.imshow.call_args[0][0]
    assert isinstance(call_args, np.ndarray)
    assert call_args.shape == (28, 28, 3)
