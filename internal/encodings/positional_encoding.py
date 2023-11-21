import torch


class PositionalEncoding(torch.nn.Module):
    def __init__(self, input_channels: int, n_frequencies: int, log_sampling: bool = True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super().__init__()
        self.n_frequencies = n_frequencies
        self.input_channels = input_channels
        self.funcs = [torch.sin, torch.cos]
        self.output_channels = input_channels * (len(self.funcs) * n_frequencies + 1)

        max_frequencies = n_frequencies - 1
        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_frequencies, steps=n_frequencies)

        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** max_frequencies, steps=n_frequencies)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]

        return torch.cat(out, -1)

    def get_output_n_channels(self) -> int:
        return self.output_channels
