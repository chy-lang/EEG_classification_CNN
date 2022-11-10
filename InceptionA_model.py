import torch
class InceptionA(torch.nn.Module):
    def __init__(self, in_channel):
        super(InceptionA, self).__init__()
        self.branch_pool = torch.nn.Conv2d(in_channel, 24, kernel_size=1)

        self.branch1x1 = torch.nn.Conv2d(in_channel, 16, kernel_size=1)

        self.branch5x5_1 = torch.nn.Conv2d(in_channel, 16, kernel_size=1)
        self.branch5x5_2 = torch.nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch3x3_1 = torch.nn.Conv2d(in_channel, 16, kernel_size=1)
        self.branch3x3_2 = torch.nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = torch.nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.pool = torch.nn.AvgPool2d(kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        branch_pool = self.pool(x)
        branch_pool = self.branch_pool(branch_pool)

        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        outputs = [branch_pool, branch1x1, branch5x5, branch3x3]
        return torch.cat(outputs, dim=1)