import torch
import torch.autograd as autograd

class ReReLu(autograd.Function):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def forward(self, input : torch.Tensor, *args, **kwargs):
        self.save_for_backward(input)
        return input - 0.5 * input.floor()

    @staticmethod
    def backward(self, grad_output : torch.Tensor, *args, **kwargs):
        input = self.saved_tensors[0]
        return grad_output * torch.ones_like(input)