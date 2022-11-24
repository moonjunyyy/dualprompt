import torch
import torch.autograd as autograd

class GauDiLU(autograd.Function):
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
    
class GauDinvLU(autograd.Function):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def forward(self, input : torch.Tensor, *args, **kwargs):
        flag = (input.floor().to(torch.int) % 2).to(torch.float)
        self.save_for_backward(flag)

        return torch.pow(torch.tensor(-1), flag) * input + flag * (input.floor() + 1)

    @staticmethod
    def backward(self, grad_output : torch.Tensor, *args, **kwargs):
        flag = self.saved_tensors[0]
        return grad_output * (flag * -2 + 1)