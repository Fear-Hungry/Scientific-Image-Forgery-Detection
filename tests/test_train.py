import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from forgeryseg.train import train_one_epoch


class _EmptyParamModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.empty = torch.nn.Parameter(torch.empty(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_train_one_epoch_rejects_empty_parameters():
    model = _EmptyParamModule()
    loader = DataLoader(TensorDataset(torch.randn(2, 3, 4, 4), torch.randn(2, 3, 4, 4)), batch_size=1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError, match=r"zero elements"):
        train_one_epoch(model, loader, criterion, optimizer, device="cpu", use_amp=False, progress=False)

