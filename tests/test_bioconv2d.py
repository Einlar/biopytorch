import torch
import torch.nn.functional as F
from biopytorch.bioconv2d import same_padding 
from typing import Tuple

def same_padding_tester(img_size : Tuple[int, int],
                        kernel_size : Tuple[int, int],
                        dilation : Tuple[int, int],
                        in_channels : int = 5,
                        out_channels : int = 5,
                        batch_size : int = 1):
    
    padding = same_padding(kernel_size, dilation)

    x = torch.randn(batch_size, in_channels, *img_size)
    filters = torch.randn(out_channels, in_channels, *kernel_size)

    auto = True
    if len(padding) == 2:
        out = F.conv2d(x, filters, dilation=dilation, padding=padding)    
    else: #Need to separate padding from conv2d. This introduces some overhead
        auto = False
        x = F.pad(x, padding) 
        out = F.conv2d(x, filters, dilation=dilation, padding='valid') 
        
    assert out.shape == (batch_size, out_channels, *img_size), f"Wrong out.shape = {out.shape} for input.shape = {x.shape}, kernel_size = {kernel_size}, dilation = {dilation}, computed padding = {padding}"
    
    print(f"Success with input_shape={img_size}, kernel_size={kernel_size}, dilation={dilation}, padding: {padding} " + ("(Symm padding)" if auto else "(Asymm padding)"))

def test_same_padding() -> None:
    #Kernel even, dilation odd
    same_padding_tester((32, 28), (6, 6), (1, 1)) 
    same_padding_tester((27, 27), (6, 8), (1, 3))
    same_padding_tester((32, 27), (10, 4), (3, 1))
    same_padding_tester((37, 52), (8, 8), (1, 3))

    #Kernel odd, dilation odd
    same_padding_tester((32, 28), (5, 3), (1, 1)) 
    same_padding_tester((27, 27), (7, 5), (1, 3))
    same_padding_tester((32, 27), (11, 7), (3, 1))
    same_padding_tester((37, 52), (3, 3), (3, 3))
    
    #Kernel even, dilation even
    same_padding_tester((10, 10), (6, 6), (2, 2)) 
    same_padding_tester((8, 18), (6, 8), (4, 2))
    same_padding_tester((62, 4), (10, 4), (2, 4))
    same_padding_tester((10, 16), (8, 8), (4, 4))
    
    #Kernel odd, dilation even
    same_padding_tester((32, 28), (5, 3), (2, 2)) 
    same_padding_tester((27, 27), (7, 5), (4, 2))
    same_padding_tester((32, 27), (11, 7), (2, 4))
    same_padding_tester((37, 52), (3, 3), (4, 4))
    


