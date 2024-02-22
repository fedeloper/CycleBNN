import torch.nn as nn
import torch
import torch.nn.functional as F

class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''
    def forward(self, input):
        self.save_for_backward(input)
        size = input.size()
        mean = torch.mean(input.abs(), 1, keepdim=True)
        input = input.sign()
        return input, mean

    def backward(self, grad_output, grad_output_mean):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

from lowbinary.modules.quantize import QConv2d

class BinConv2d(nn.Module):
    def __init__(self, input_channels, output_channels,
            kernel_size=-1, stride=-1, padding=-1, dropout=0,bias = False):
        super(BinConv2d, self).__init__()
        self.layer_type = 'BinConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout

        self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
        self.bn.weight.data = self.bn.weight.data.zero_().add(1.0)
        if dropout!=0:
            self.dropout = nn.Dropout(dropout)
        from modules.quantize import QConv2d
        self.conv = QConv2d(input_channels, output_channels,
                kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x,num_bits, num_grad_bits):
        x = self.bn(x)
        x, mean = BinActive.apply(x)
        if self.dropout_ratio!=0:
            x = self.dropout(x)
        x = self.conv(x,num_bits, num_grad_bits)
        x = self.relu(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()     
        self.l1 =   QConv2d(3, 192, kernel_size=5, stride=1, padding=2)
        self.l2 = nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False)
        self.l3 = nn.ReLU(inplace=False)
        self.l4 = BinConv2d(192, 160, kernel_size=1, stride=1, padding=0)
        self.l5=  BinConv2d(160,  96, kernel_size=1, stride=1, padding=0)
        self.l6 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.l7=  BinConv2d( 96, 192, kernel_size=5, stride=1, padding=2, dropout=0.5)
        self.l8 = BinConv2d(192, 192, kernel_size=1, stride=1, padding=0)
        self.l9 = BinConv2d(192, 192, kernel_size=1, stride=1, padding=0)
        self.l10=  nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.l11 = BinConv2d(192, 192, kernel_size=3, stride=1, padding=1, dropout=0.5)
        self.l12 = BinConv2d(192, 192, kernel_size=1, stride=1, padding=0)
        self.l13 = nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False)
        self.l14 = QConv2d(192,  10, kernel_size=1, stride=1, padding=0)
        self.l15 = nn.ReLU(inplace=False)
        self.l16 = nn.AvgPool2d(kernel_size=8, stride=1, padding=0)
          

    def forward(self, x,num_bits, num_grad_bits):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.clamp_(min=0.01)
        #x = self.xnor(x)
        x = self.l1(x,num_bits, num_grad_bits)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x,num_bits, num_grad_bits)
        x = self.l5(x,num_bits, num_grad_bits)
        x = self.l6(x)
        x = self.l7(x,num_bits, num_grad_bits)
        x = self.l8(x,num_bits, num_grad_bits)
        x = self.l9(x,num_bits, num_grad_bits)
        x = self.l10(x)
        x = self.l11(x,num_bits, num_grad_bits)
        x = self.l12(x,num_bits, num_grad_bits)
        x = self.l13(x)
        x = self.l14(x,num_bits, num_grad_bits)
        x = self.l15(x)
        x = self.l16(x)
        x = x.view(x.size(0), 10)
        return x
