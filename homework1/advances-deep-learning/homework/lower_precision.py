from pathlib import Path
import torch
from .bignet import BIGNET_DIM, LayerNorm

def block_quantize_3bit_stream(x: torch.Tensor, group_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 1
    assert x.size(0) % group_size == 0

    x = x.view(-1, group_size)
    normalization = x.abs().max(dim=-1, keepdim=True).values

    # Scale to [0, 7]
    x_norm = (x + normalization) / (2 * normalization)
    q = (x_norm * 7).round().to(torch.int16).clamp(0, 7)
    
    num_blocks = x.size(0)
    packed = torch.empty((num_blocks, 12), dtype=torch.int8, device=x.device)
    
    # Pack 32 elements per block. We process in 2 halves of 16 elements using int64 shifting.
    # First 16 elements -> first 6 bytes
    shifted1 = torch.zeros(num_blocks, dtype=torch.int64, device=x.device)
    for i in range(16):
        shifted1 |= q[:, i].to(torch.int64) << (i * 3)
        
    # Second 16 elements -> next 6 bytes
    shifted2 = torch.zeros(num_blocks, dtype=torch.int64, device=x.device)
    for i in range(16):
        shifted2 |= q[:, 16 + i].to(torch.int64) << (i * 3)
        
    for i in range(6):
        packed[:, i] = (shifted1 >> (i * 8)) & 0xFF
        packed[:, 6 + i] = (shifted2 >> (i * 8)) & 0xFF
        
    return packed, normalization.to(torch.float16)


def block_dequantize_3bit_stream(packed: torch.Tensor, normalization: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    assert packed.dim() == 2
    normalization = normalization.to(torch.float32)
    num_blocks = packed.size(0)
    
    shifted1 = torch.zeros(num_blocks, dtype=torch.int64, device=packed.device)
    shifted2 = torch.zeros(num_blocks, dtype=torch.int64, device=packed.device)
    
    for i in range(6):
        shifted1 |= (packed[:, i].to(torch.int64) & 0xFF) << (i * 8)
        shifted2 |= (packed[:, 6 + i].to(torch.int64) & 0xFF) << (i * 8)
        
    q = torch.empty((num_blocks, group_size), dtype=torch.float32, device=packed.device)
    for i in range(16):
        q[:, i] = (shifted1 >> (i * 3)) & 0x7
        q[:, 16 + i] = (shifted2 >> (i * 3)) & 0x7
    
    # Dequantize to float32
    x_norm = q / 7.0
    x = (x_norm * 2 * normalization) - normalization
    return x.view(-1)


class Linear3BitStream(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 32) -> None:
        super().__init__()
        self._shape = (out_features, in_features)
        self._group_size = group_size

        total_elements = out_features * in_features
        num_blocks = total_elements // group_size

        # 32 params map to exactly 12 bytes of tensor memory allocation
        self.register_buffer(
            "weight_q3",
            torch.zeros(num_blocks, 12, dtype=torch.int8),
            persistent=False,
        )
        self.register_buffer(
            "weight_norm",
            torch.zeros(num_blocks, 1, dtype=torch.float16),
            persistent=False,
        )
        
        self._register_load_state_dict_pre_hook(Linear3BitStream._load_state_dict_pre_hook, with_module=True)
        
        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float32))

    def _load_state_dict_pre_hook(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if f"{prefix}weight" in state_dict:
            weight = state_dict[f"{prefix}weight"].reshape(-1)
            del state_dict[f"{prefix}weight"]
            weight_q3, weight_norm = block_quantize_3bit_stream(weight, self._group_size)
            self.weight_q3.copy_(weight_q3)
            self.weight_norm.copy_(weight_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            dequant_weight = block_dequantize_3bit_stream(self.weight_q3, self.weight_norm, self._group_size)
            dequant_weight = dequant_weight.view(self._shape)
            return torch.nn.functional.linear(x, dequant_weight, self.bias)


class BigNet3BitStream(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            # group_size 32 provides optimal accuracy vs metadata-footprint balancing
            self.model = torch.nn.Sequential(
                Linear3BitStream(channels, channels, group_size=32),
                torch.nn.ReLU(),
                Linear3BitStream(channels, channels, group_size=32),
                torch.nn.ReLU(),
                Linear3BitStream(channels, channels, group_size=32),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> BigNet3BitStream:
    net = BigNet3BitStream()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net