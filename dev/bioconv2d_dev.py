#This files contains functions to do the BioConv2d + logging everything, used mainly for debugging

import torch
import torch.nn.functional as F

from biopytorch.bioconv2d import same_padding

import logging
logging.basicConfig(filename='bioconv2d_dev.log', encoding='utf-8', level=logging.INFO, filemode='w')

from typing import Union, Tuple

_size_2_t = Union[int, Tuple[int, int]] #nn.common_types

def _size_2_t_to_tuple(arg : _size_2_t) -> Tuple[int, int]:
    """Converts an int `arg` to a 2-tuple of equal entries `(arg, arg)`. If `arg` is already a 2-tuple, nothing is done.

    Examples
    --------
    1      -> (1, 1)
    (3,)   -> (3, 3)
    (1, 2) -> (1, 2)
    """
    
    try:
        arg_tuple = tuple(arg)
    except TypeError:
        arg_tuple = tuple([arg])
    
    if len(arg_tuple) != 2:
        return ( int(arg_tuple[0]), ) * 2
        
    return ( int(arg[0]), int(arg[1]) )

def update_filters(img : torch.Tensor,
                   filters : torch.Tensor,
                   lebesgue_p : int = 2,
                   delta : float = 0.4,
                   ranking_param : int = 2,
                   stride : Union[int, Tuple[int]] = 1,
                   padding : Union[str, Union[int, Tuple[int, int]]] = 0,
                   dilation = 1,
                   groups = 1,
                   padding_mode = 'zeros',
                   ) -> torch.Tensor:
    """
    Fast "BioLearning" rule from "Unsupervised Learning by Competing Hidden Units" by Krotov & Hopfield (https://arxiv.org/pdf/1806.10181v2.pdf) applied to a Convolutional Layer.
    
    Parameters
    ----------
    img : torch.Tensor
        Input of the convolutional layer. Shape is (minibatch, channels, height, width).
        For instance, a batch of 64 32x32 RGB images has shape (64, 3, 32, 32).
    filters : torch.Tensor
        Weights for the convolutional kernels. Shape is (out_channels, in_channels, kernel_height, kernel_width).
    lebesgue_p : int, 2 by default
    delta : float, 0.4 by default
    ranking_param : int, 2 by default
    stride : int, 1 by default 
        Stride for the convolutional kernels (see torch.nn.Conv2d docs)
    padding : int, 0 by default
        Padding for the convolutional kernels (see torch.nn.Conv2d docs)
        
    Note: both `img` and `filters` should be of the same *dtype* and on the same *device*.
        
    Returns
    -------
    delta_weights : torch.Tensor
        Computed change of weights by one step of Hebbian learning. It is normalized to have `torch.max(delta_weights) == 1`.
    """
    
    logging.info("--------------Update filters---------------")
    
    logging.info(f"Received img.shape = {img.shape} (minibatch, channels, height, width) \nfilters.shape = {filters.shape} (out_channels, in_channels, kernel_height, kernel_width)")
    
    in_channels = img.size(1)
    out_channels = filters.size(0)
    assert out_channels // groups > 1, f"There must be more than one kernel per group to have competition. The number of output channels ({out_channels}) must be a higher multiple of the number of groups ({groups})."
    
    assert in_channels % groups == 0, f"The input has {in_channels} which is not divisible by {groups}"
    assert out_channels % groups == 0, f"The output channels are {out_channels}, which is not divisible by {groups}"
    
    if groups > 1:
        img_groups = torch.chunk(img, groups, dim=1) 
        filter_groups = torch.chunk(filters, groups, dim=0)
        
        return torch.cat([
            update_filters(img=x,
                           filters=f,
                           lebesgue_p=lebesgue_p,
                           delta=delta,
                           ranking_param=ranking_param,
                           stride=stride,
                           padding=padding,
                           dilation=dilation,
                           groups=1,
                           padding_mode=padding_mode)
            for x, f in zip(img_groups, filter_groups)
        ], dim = 0)
        
    #img.shape     = (minibatch, channels, height, width)
    
    out_channels, in_channels, kernel_height, kernel_width = filters.shape
    
    weights = filters.reshape(out_channels, -1) 
    #weights.shape = (out_channels, n_parameters_per_kernel)
    #There are `out_channels` filters (kernels) that are applied to the input, each with weights.shape[-1] parameters, one for each pixel in their local receptive field.
    #Kernels are "applied" (with a scalar product) to "patches" of the image, each of the shape of the local receptive field. 
    
    dilation = _size_2_t_to_tuple(dilation)
    kernel_size = (kernel_height, kernel_width)
    
    img, added_padding, new_padding = preliminary_padding(img, padding, kernel_size, dilation, padding_mode)
    
    # if padding == 'valid':
    #     padding = 0
    # if padding == 'same':
        
    #     padding = same_padding((kernel_height, kernel_width), dilation)
    
    #     if len(padding) == 4:
    #         img = F.pad(img, padding, mode=padding_mode)
    #         padding = (0, 0) #No more padding needed
    
    # padding = _size_2_t_to_tuple(padding)
    
    # if padding_mode != 'zeros':
    #     pad_height, pad_width = padding
    #     img = F.pad(img, (pad_width, pad_width, pad_height, pad_height), mode=padding_mode)
    #     padding = (0, 0)
    
    patches = F.unfold(img, kernel_size=(kernel_height, kernel_width), stride=stride, padding=new_padding, dilation=dilation) #Extract all the patches for the input images
    #Shape is (minibatch, n_parameters_per_kernel, n_patches)
    logging.info(f"patches.shape = {patches.shape} (minibatch, n_parameters_per_kernel, n_patches)")
    n_parameters_per_kernel = patches.size(1)
    patches = patches.swapaxes(0, 1).reshape(n_parameters_per_kernel, -1) 
    #Shape is now (n_parameters_per_kernel, n_all_patches_in_minibatch)
    logging.info(f"patches.shape = {patches.shape} (n_parameters_per_kernel, n_all_patches_in_minibatch)")
    
    currents = torch.matmul(torch.sign(weights) * torch.abs(weights) ** (lebesgue_p - 1), patches) #"Manual" conv2d using a Lebesgue p-norm. Basically, each kernel (total: `out_channel`) is applied to a patch (total: `n_patches`), producing a scalar activation.
    #Shape is (out_channels, n_all_patches_in_minibatch)
    logging.info(f"currents.shape = {currents.shape} (out_channels, n_all_patches_in_minibatch)")
    
    #currents = currents.view(-1, currents.shape[-1])

    _, ranking_indices = currents.topk(ranking_param, dim=0) #ranking_indices[k][i] contains the index of the kernel with the k-highest activation when applied to the i-th patch.
    #Shape is (ranking_param, n_all_patches_in_minibatch)
    logging.info(f"ranking_indices.shape = {ranking_indices.shape} (ranking_param, n_all_patches_in_minibatch)")
    
    post_activation_currents = torch.zeros_like(currents)
    #Shape is (out_channels, n_all_patches_in_minibatch)
    batch_indices = torch.arange(patches.size(-1))
    post_activation_currents[ranking_indices[0], batch_indices] = 1.0 #The kernels that are most active for a patch are updated towards it
    post_activation_currents[ranking_indices[ranking_param - 1], batch_indices] = -delta #Others that are less activated are pushed in the opposite direction (i.e. they will be "less active" when encountering a similar patch in the future)
    logging.info(f"post_activation_currents.shape = {post_activation_currents.shape} (out_channels, n_all_patches_in_minibatch)")

    delta_weights = torch.matmul(post_activation_currents, torch.t(patches))
    #Shape is (out_channels, n_parameters_per_kernel)
    logging.info(f"delta_weights.shape = {delta_weights.shape} (out_channels, n_parameters_per_kernel)")
    
    second_term = torch.sum(torch.mul(post_activation_currents, currents), dim=1)
    #Shape is (out_channel,)
    logging.info(f"second_term.shape = {second_term.shape} (out_channel)")
    delta_weights.sub_(second_term.unsqueeze(1) * weights)
    
    #Normalize
    nc = torch.max(torch.abs(delta_weights))
    delta_weights.div_(nc)
    
    return delta_weights.reshape(filters.shape)

def preliminary_padding(x : torch.Tensor,
                        padding : Union[str, _size_2_t],
                        kernel_size : _size_2_t,
                        dilation : _size_2_t,
                        padding_mode : str = 'zeros',
                       ) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Apply to a tensor `x` any padding that cannot be applied directly by F.conv2d (e.g. asymmetric padding, or when padding_mode is not 'zeros').
    
    Returns
    -------
    x_with_padding : torch.Tensor
        The input tensor with (eventually) the added preliminary padding.
    added_padding : Tuple[int, int, int, int]
        Padding that has been applied
    remaining_padding : Tuple[int, int]
        Padding that can be applied by F.conv2d. This is used when the padding is symmetric with 'zeros', meaning that it is not necessary to use F.pad, which is slow, but directly F.conv2d.
    """
    
    symmetric = True 
    
    if isinstance(padding, str):
        if padding == 'valid':
            padding = 0
        elif padding == 'same':
            padding = same_padding(kernel_size, dilation)
            
            if len(padding) == 4:
                symmetric = False #Asymmetric padding is needed
        else:
            raise NotImplementedError(f'"{padding}" is not implemented.')
    
    if symmetric:
        padding = _size_2_t_to_tuple(padding) 
        
        if padding_mode != 'zeros': #F.pad must be used
            pad_height, pad_width = padding
            x = F.pad(x, (pad_width, pad_width, pad_height, pad_height), mode=padding_mode)
            
            return x, (pad_width, pad_width, pad_height, pad_height), (0, 0) 
        
        return x, (0, 0, 0, 0), padding
        
    else: #Asymmetric must be manual
        if padding_mode == 'zeros':
            padding_mode = 'constant' #because F.pad uses a different convention...
        x = F.pad(x, padding, mode=padding_mode)
        
        return x, padding, (0, 0)

def update_filters_conv(img : torch.Tensor,
                        filters : torch.Tensor,
                        lebesgue_p : int = 2,
                        delta : float = 0.4,
                        ranking_param : int = 2,
                        stride : Union[int, Tuple[int]] = 1,
                        padding : Union[str, _size_2_t] = 0,
                        dilation = 1,
                        padding_mode : str = 'zeros',
                        groups : int = 1) -> torch.Tensor:
    """
    Fast "BioLearning" rule from "Unsupervised Learning by Competing Hidden Units" by Krotov & Hopfield (https://arxiv.org/pdf/1806.10181v2.pdf) applied to a Convolutional Layer.
    
    Parameters
    ----------
    img : torch.Tensor
        Input of the convolutional layer. Shape is (minibatch, channels, height, width).
        For instance, a batch of 64 32x32 RGB images has shape (64, 3, 32, 32).
    filters : torch.Tensor
        Weights for the convolutional kernels. Shape is (out_channels, in_channels, kernel_height, kernel_width).
    lebesgue_p : int, 2 by default
    delta : float, 0.4 by default
    ranking_param : int, 2 by default
    stride : int, 1 by default 
        Stride for the convolutional kernels (see torch.nn.Conv2d docs)
    padding : int, 0 by default
        Padding for the convolutional kernels (see torch.nn.Conv2d docs)
        
    Note: both `img` and `filters` should be of the same *dtype* and on the same *device*.
        
    Returns
    -------
    delta_weights : torch.Tensor
        Computed change of weights by one step of Hebbian learning. It is normalized to have `torch.max(delta_weights) == 1`.
    """
    
    logging.info("--------------Update filters conv---------------")
    logging.info(f"Received img.shape = {img.shape} and filters.shape = {filters.shape}")
    
    kernel_size = (filters.size(-2), filters.size(-1))
    dilation = _size_2_t_to_tuple(dilation)
    
    
    if groups > 1:
        img_groups = torch.chunk(img, groups, dim=1) 
        filter_groups = torch.chunk(filters, groups, dim=0)
        
        return torch.cat([
            update_filters_conv(img=x,
                           filters=f,
                           lebesgue_p=lebesgue_p,
                           delta=delta,
                           ranking_param=ranking_param,
                           stride=stride,
                           padding=padding,
                           dilation=dilation,
                           groups=1,
                           padding_mode=padding_mode)
            for x, f in zip(img_groups, filter_groups)
        ], dim = 0)

    #TODO stride > 1 and 'same' do not work!
    img, added_padding, new_padding = preliminary_padding(img, padding, kernel_size, dilation, padding_mode)
    
    filters_p = torch.sign(filters) * torch.abs(filters) ** (lebesgue_p - 1)
    currents = F.conv2d(img, filters_p, stride=stride, padding=new_padding, dilation=dilation)
    #Apply each kernel (total: channel_out) to each "patch" in the image
    batch_size, out_channels, height_out, width_out = currents.shape
    logging.info(f"currents.shape = {currents.shape})")
    
    currents = currents.transpose(0, 1).reshape(out_channels, -1)
    #Shape is now (out_channels, n_all_patches_in_minibatch)    
    logging.info(f"currents.shape = {currents.shape})")
    
    _, ranking_indices = currents.topk(ranking_param, dim=0)
    #Shape is (ranking_param, n_all_patches_in_minibatch)
    logging.info(f"ranking_indices.shape = {ranking_indices.shape}")
    
    post_activation_currents = torch.zeros_like(currents)
    #Shape is (out_channels, n_all_patches_in_minibatch)
    
    batch_indices = torch.arange(currents.size(-1))
    post_activation_currents[ranking_indices[0], batch_indices] = 1.0 #The kernels that are most active for a patch are updated towards it
    post_activation_currents[ranking_indices[ranking_param - 1], batch_indices] = -delta #Others that are less activated are pushed in the opposite direction (i.e. they will be "less active" when encountering a similar patch in the future)
    
    logging.info(f"post_activation_currents.shape = {post_activation_currents.shape}")
    
    #We want to weight each patch in the minibatch by the corresponding post_activation_current,
    #and compute their weighted sum. This is done by using a 2d convolution with a specially designed filter, such that each entry in `activation_filter` is applied exactly to the entries of a single patch
    
    #For instance, consider the following 2D tensor:
    # 0 1 2 2
    # 1 0 1 0
    # 2 1 3 1
    #The 2x2 patches (stride = 1, padding = 0) are:
    # 0 1 | 1 2 | 2 2 | 1 0 | 0 1 | 1 0
    # 1 0 | 0 1 | 1 0 | 2 1 | 1 3 | 3 1
    #And we want to weight them with:
    #  1     0     1     2     1     0
    #which gives:
    # 4 4
    # 7 5
    #This can be done by convolving the original 2D tensor with the weights rearranged as a 2x3 filter:
    # 1 0 1
    # 2 1 0
    # Focus on just one entry of this filter, e.g. the first "1" (at (1, 1) position). When the filter is applied during the 2d convolution, this "1" overlaps exactly with the first patch, i.e.
    # 0 1
    # 1 0
    # which is thus "weighted" by this "1". 
    
    #This schema can be applied also with stride, by adding 0s to the filter. For instance, consider again
    # 0 1 2 2
    # 1 0 1 0
    # 2 1 3 1
    #The 2x2 patches (stride = 2, padding = 0) are:
    # 0 1 | 2 2
    # 1 0 | 1 0
    #Suppose the patch weights are:
    #  2     3
    #The idea is to add 0s between them, so that they "act" on the correct patches. In this case:
    # 2 0 3
    # 0 0 0 
    #In the general case, this can be constructed by constructing a block matrix, where each block has a unique non-zero value in the top-left given by a weight, and the block size is the stride. So, for a (2, 2) stride, we have 2x2 blocks:
    # 2 -> 2 0 
    #      0 0
    #For stride = (3, 3):
    # 5 -> 5 0 0
    #      0 0 0
    #      0 0 0
    #Then, the [2, 3] weights for stride=(2,2) become:
    # 2 0 3 0
    # 0 0 0 0
    #Which must be "cropped" to the correct size. We know that, after the 2d conv, the resulting shape should be that of the filters. So we can invert the formula for the output shape of F.conv_2d to find the correct dimensions for the kernel, which in this case are 2x3.
    
    #In the case of batches, we use the batch dimension as the channels one. It works.
    
    separated_images = img.transpose(0, 1) #Shape is (in_channels, batch_size, height, width)
    activation_filter = post_activation_currents.view(out_channels, batch_size, height_out, width_out)
    
    #stride can be a tuple (stride_x, stride_y) or an integer
    try:
        stride = tuple(stride)
    except TypeError:
        stride = tuple([stride])
    
    if len(stride) != 2:
        stride = (stride[0],) * 2
    
    stride_mask = torch.zeros(stride[0], stride[1], device=img.device, dtype=img.dtype)    
    stride_mask[0][0] = 1
    activation_filter = torch.kron(activation_filter, stride_mask) #construct a block matrix
    
    #padding can be a tuple (padding_x, padding_y) or an integer
    
    filter_dims = (img.size(-2) + 2 * new_padding[0] + 1 - kernel_size[0],
                   img.size(-1) + 2 * new_padding[1] + 1 - kernel_size[1])
    
    logging.info(f"filter_dims: {filter_dims}")
    activation_filter = activation_filter[..., :filter_dims[0], :filter_dims[1]]
    
    logging.info(f"separated_images.shape = {separated_images.shape}")
    logging.info(f"activation_filter.shape = {activation_filter.shape}")
    
    delta_weights = F.conv2d(separated_images, activation_filter, padding=new_padding, stride=dilation).swapaxes(0, 1) #setting stride=dilation gives for free support for the dilation argument, in the sense that it gives equal results with the unfold-version code. However, this should be checked a bit better, as I don't have a full theoretical understanding on why this works in practice.
    
    #Shape is (out_channels, in_channels, kernel_height, kernel_width)
    logging.info(f"delta_weights.shape = {delta_weights.shape}")
    
    second_term = torch.sum(torch.mul(post_activation_currents, currents), dim=1)
    #Shape is (out_channels,)
    logging.info(f"second_term.shape = {second_term.shape}")
    delta_weights.sub_(second_term.view(-1, 1, 1, 1) * filters)
    
    nc = torch.max(torch.abs(delta_weights))
    delta_weights.div_(nc)
    
    return delta_weights


# images = torch.cat((cifar10.train_dataset[50][0].unsqueeze(0), cifar10.train_dataset[51][0].unsqueeze(0))) #batch of 2 images
# filters = torch.randn(5, 3, 7, 7) #5 7x7 RGB filters

# update_filters_conv(images, filters);