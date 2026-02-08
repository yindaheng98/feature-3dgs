import torch
import torch.nn as nn
from feature_3dgs.decoder import AbstractDecoder

class CNNDecoder(AbstractDecoder):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=1).cuda()

    def load_checkpoint(self, path: str | None = None) -> None:
        if path is None:
            return
        else:
            self.conv = torch.load(path)

    def __call__(self, x: torch.Tensor):
        return self.conv(x)

    def save(self, save_path: str):
        with torch.no_grad():
            torch.save(self.conv.weight)