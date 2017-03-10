from timeit import timeit
from termcolor import colored as clr


fwd_setup = """
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

torch.set_num_threads(4)

class TestNet(nn.Module):

    def __init__(self, in_size, out_size):
        super(TestNet, self).__init__()

        self.l1 = nn.Linear(in_size, in_size)
        self.l2 = nn.ReLU()
        self.l3 = nn.Linear(in_size, in_size)
        self.l4 = nn.ReLU()
        self.l5 = nn.Linear(in_size, out_size)
        self.l6 = nn.Tanh()
        self.l7 = nn.Linear(out_size, out_size)
        self.layers = \
                [self.l1, self.l2, self.l3, self.l4, self.l5, self.l6, self.l7]

    def forward(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

batch_size = {:d}
in_size    = {:d}
out_size   = {:d}
device     = '{:s}'

net = TestNet(in_size, out_size)
inputs = torch.randn(batch_size, in_size)

if device == 'GPU':
    net.cuda()
    inputs = inputs.cuda()
    torch.cuda.synchronize()

inputs = Variable(inputs)
"""


bwd_setup = fwd_setup + """

outputs = net(inputs)
targets = Variable(torch.rand(outputs.size()))

if device == 'GPU':
    targets = targets.cuda()

net.zero_grad()
criterion = nn.MSELoss()
loss = criterion(outputs, targets)

if device == 'GPU':
    torch.cuda.synchronize()
"""

if __name__ == "__main__":
    out_size = 128
    in_sizes = [256, 2048]
    batch_sizes = [32, 128, 256]
    print("---------- FORWARD ----------")
    for device in ["CPU", "GPU"]:
        total = .0
        for in_size in in_sizes:
            for batch_size in batch_sizes:
                t = timeit(
                    ("net(inputs);\n"
                     "if device == 'GPU':\n"
                     "    torch.cuda.synchronize()"),
                    setup=fwd_setup.format(batch_size,in_size,out_size,device),
                    number=100
                )
                total += t
                """
                print("forward @{:s} {:4d}x({:4d} -> {:4d}) ===> {:f}".format(
                    device, batch_size, in_size, out_size, t
                ))
                """
        print(clr("FORWARD", "yellow") +
              " @ {:s} ===> {:s} s.".format(clr(device, "yellow"),
                                         clr("{:.4f}".format(total), "red")
              ))

    print("---------- BACKWARD ----------")
    for device in ["CPU", "GPU"]:
        total = .0
        for in_size in in_sizes:
            for batch_size in batch_sizes:
                t = timeit(
                    ("loss.backward(retain_variables=True);\n"
                     "if device == 'GPU':"
                     "    torch.cuda.synchronize()"),
                    setup=bwd_setup.format(batch_size,in_size,out_size,device),
                    number=100
                )
                total += t
                """
                print("backward @{:s} {:4d}x({:4d} -> {:4d}) ===> {:f}".format(
                    device, batch_size, in_size, out_size, t
                ))
                """
        print(clr("BACKWARD", "yellow") +
              " @ {:s} ===> {:s} s.".format(clr(device, "yellow"),
                                         clr("{:.4f}".format(total), "red")
              ))
