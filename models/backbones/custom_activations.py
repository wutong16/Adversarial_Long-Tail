# Imports

# Import basic libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import OrderedDict

# Import PyTorch
import torch # import main library
from torch.autograd import Variable
import torch.nn as nn # import modules
from torch.autograd import Function # import Function to create custom activations
from torch.nn.parameter import Parameter # import Parameter to create custom activations with learnable parameters
from torch import optim # import optimizers for demonstrations
import torch.nn.functional as F # import torch functions
from torchvision import datasets, transforms # import transformations to use for demo

def build_custom_activation(name='relu', **kwargs):
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'silu':
        return SiLU()
    elif name == 'soft_exp':
        alpha = kwargs.get('alpha', 0)
        return soft_exponential(alpha)
    elif name == 'brelu':
        raise NotImplementedError
    else:
        raise NameError

# simply define a silu function
def silu(input):
    '''
    Applies the Sigmoid Linear Unit (SiLU) function element-wise:
        SiLU(x) = x * sigmoid(x)
    '''
    return input * torch.sigmoid(input)

class SiLU(nn.Module):

    def __init__(self):
        super().__init__() #

    def forward(self, input):

        return silu(input)

class soft_exponential(nn.Module):

    def __init__(self, alpha = None):

        super(soft_exponential,self).__init__()

        # initialize alpha
        if alpha == None:
            self.alpha = Parameter(torch.tensor(0.0))
        else:
            self.alpha = Parameter(torch.tensor(alpha))

        self.alpha.requiresGrad = True

    def forward(self, x):

        if (self.alpha == 0.0):
            return x

        if (self.alpha < 0.0):
            return - torch.log(1 - self.alpha * (x + self.alpha)) / self.alpha

        if (self.alpha > 0.0):
            return (torch.exp(self.alpha * x) - 1)/ self.alpha + self.alpha



class brelu(Function):
    '''
        https://arxiv.org/pdf/1709.04054.pdf
    '''
    #both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input) # save input for backward pass

        # get lists of odd and even indices
        input_shape = input.shape[0]
        even_indices = [i for i in range(0, input_shape, 2)]
        odd_indices = [i for i in range(1, input_shape, 2)]

        # clone the input tensor
        output = input.clone()

        # apply ReLU to elements where i mod 2 == 0
        output[even_indices] = output[even_indices].clamp(min=0)

        # apply inversed ReLU to inversed elements where i mod 2 != 0
        output[odd_indices] = 0 - output[odd_indices] # reverse elements with odd indices
        output[odd_indices] = - output[odd_indices].clamp(min = 0) # apply reversed ReLU

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        grad_input = None # set output to None

        input, = ctx.saved_tensors # restore input from context

        # check that input requires grad
        # if not requires grad we will return None to speed up computation
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.clone()

            # get lists of odd and even indices
            input_shape = input.shape[0]
            even_indices = [i for i in range(0, input_shape, 2)]
            odd_indices = [i for i in range(1, input_shape, 2)]

            # set grad_input for even_indices
            grad_input[even_indices] = (input[even_indices] >= 0).float() * grad_input[even_indices]

            # set grad_input for odd_indices
            grad_input[odd_indices] = (input[odd_indices] < 0).float() * grad_input[odd_indices]

        return grad_input

if __name__ == '__main__':
    pass