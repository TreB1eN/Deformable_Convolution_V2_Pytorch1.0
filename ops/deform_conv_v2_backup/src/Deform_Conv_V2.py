import math
from torch import nn
from torch.autograd import Function
import torch

import modulated_deform_conv_cuda
from torch.nn import init

class DeformConvV2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1), padding=(0,0), dilation=(1,1), use_bias=False):
        super(DeformConvV2, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels * kernel_size[0] * kernel_size[1]))
        if use_bias:
            self.use_bias = True
            self.bias = nn.Parameter(torch.Tensor(out_channels, 1))
        else:
            self.use_bias = False
        self.reset_parameters()
        self.img2col = DeformConvImg2Col(in_channels, out_channels, kernel_size, stride, padding, dilation)
    
    def reset_parameters(self):
        init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu')
        if self.use_bias:
            init.constant_(self.bias, 0)
        
    def forward(self, data_im):
        input_shape = data_im.shape
        data_col = self.img2col(data_im)
        data_col = data_col.view(data_col.shape[0], -1)
        data_out = torch.mm(self.weight, data_col)
        if self.use_bias:
            data_out += self.bias
        data_out = data_out.view(self.weight.shape[0], input_shape[1], input_shape[2], input_shape[3])
        return data_out.transpose(1,0)
    

class DeformConvImg2Col(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1), padding=(0,0), dilation=(1,1)):
        super(DeformConvImg2Col, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
    
    def _output_size(self, input):
        output_size = (input.size(0), self.out_channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = self.padding[d]
            kernel = self.dilation[d] * (self.kernel_size[d] - 1) + 1
            stride = self.stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride + 1, )
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                "convolution input is too small (output would be {})".format(
                        'x'.join(map(str, output_size))))
        return output_size

    def forward(self, data_im, offset, mask):
        output_size = self._output_size(data_im)
        return DeformConvImg2ColFunction.apply(data_im, offset, mask, self.in_channels, output_size, self.kernel_size, self.stride, self.padding, self.dilation)

class DeformConvImg2ColFunction(Function):
    @staticmethod
    def forward(ctx, data_im, offset, mask, in_channels, output_size, kernel_size, stride, padding, dilation):
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation     

        data_col = torch.zeros([in_channels * kernel_size[0] * kernel_size[1], output_size[0], output_size[2], output_size[3]], dtype=torch.float32).to(data_im.device)

        if data_im.is_cuda and offset.is_cuda and mask.is_cuda:
            modulated_deform_conv_cuda.forward(data_im, offset, 
                                                mask, data_col, 
                                                kernel_size[0], kernel_size[1], 
                                                padding[0], padding[1], 
                                                stride[0], stride[1], 
                                                dilation[0], dilation[1])
            ctx.save_for_backward(data_im, offset, mask)
        else:
            raise NotImplementedError

        return data_col

    @staticmethod
    def backward(ctx, grad_col):
        [data_im, offset, mask] = ctx.saved_tensors

        grad_im = None
        grad_offset = None
        grad_mask = None
        if ctx.needs_input_grad[0]:
            grad_im = torch.zeros_like(data_im, dtype=torch.float32).to(data_im.device)
            grad_offset = torch.zeros_like(offset, dtype=torch.float32).to(offset.device)
            grad_mask = torch.zeros_like(mask, dtype=torch.float32).to(mask.device)

            modulated_deform_conv_cuda.backward(data_im, offset, mask, grad_col, 
                                                ctx.kernel_size[0], ctx.kernel_size[1], 
                                                ctx.padding[0], ctx.padding[1], 
                                                ctx.stride[0], ctx.stride[1], 
                                                ctx.dilation[0], ctx.dilation[1],
                                                grad_offset, grad_mask, grad_im)

        return grad_im, grad_offset, grad_mask, None, None, None, None, None, None, None