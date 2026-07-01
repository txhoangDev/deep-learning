from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image

from .autoregressive import Autoregressive
from .bsq import Tokenizer


class Compressor:
    def __init__(self, tokenizer: Tokenizer, autoregressive: Autoregressive):
        super().__init__()
        self.tokenizer = tokenizer
        self.autoregressive = autoregressive
        self.precision = 16
        self.frequencies = 1 << self.precision

    def compress(self, x: torch.Tensor) -> bytes:
        """
        Compress the image into a torch.uint8 bytes stream (1D tensor).

        Use arithmetic coding.
        """
        self.tokenizer.eval()
        self.autoregressive.eval()
        device = next(self.autoregressive.parameters()).device

        if x.ndim == 3:
            x = x.unsqueeze(0)

        with torch.no_grad():
            tokens = self.tokenizer.encode_index(x.to(device)).long()
            logits, _ = self.autoregressive(tokens)
            probs = torch.nn.functional.softmax(logits, dim=-1)

        tokens_flat = tokens[0].flatten().tolist()
        probs_flat = probs[0].flatten(0, 1)

        low = 0
        width = 1

        for i, sym in enumerate(tokens_flat):
            p = (probs_flat[i].detach().cpu() * self.frequencies).long()
            p = torch.clamp(p, min=1)

            diff = self.frequencies - int(p.sum())
            if diff > 0:
                p[p.argmax()] += diff
            elif diff < 0:
                for _ in range(-diff):
                    idx = p.argmax()
                    if p[idx] > 1:
                        p[idx] -= 1

            cdf = torch.zeros_like(p)
            cdf[1:] = torch.cumsum(p[:-1], dim=0)

            low = (low << self.precision) + width * int(cdf[sym])
            width *= int(p[sym])

        total_bits = self.precision * len(tokens_flat)
        for byte_len in range(1, total_bits // 8 + 1):
            shift = total_bits - 8 * byte_len
            unit = 1 << shift
            point = ((low + unit - 1) // unit) * unit

            if point < low + width:
                return (point >> shift).to_bytes(byte_len, "big")

        return low.to_bytes(total_bits // 8, "big")

    def decompress(self, x: bytes) -> torch.Tensor:
        """
        Decompress a tensor into a PIL image.
        You may assume the output image is 150 x 100 pixels.
        """
        h, w = 20, 30
        N = h * w

        self.tokenizer.eval()
        self.autoregressive.eval()
        device = next(self.autoregressive.parameters()).device
        tokens = torch.zeros((1, h, w), dtype=torch.long, device=device)

        total_bits = self.precision * N
        if len(x) == 0 or len(x) * 8 > total_bits:
            raise ValueError("Invalid compressed stream")

        V = int.from_bytes(x, "big") << (total_bits - len(x) * 8)

        with torch.no_grad():
            for i in range(h):
                for j in range(w):
                    logits, _ = self.autoregressive(tokens)
                    probs = torch.nn.functional.softmax(logits[0, i, j], dim=-1)

                    p = (probs.detach().cpu() * self.frequencies).long()
                    p = torch.clamp(p, min=1)

                    diff = self.frequencies - int(p.sum())
                    if diff > 0:
                        p[p.argmax()] += diff
                    elif diff < 0:
                        for _ in range(-diff):
                            idx = p.argmax()
                            if p[idx] > 1:
                                p[idx] -= 1

                    cdf = torch.zeros_like(p)
                    cdf[1:] = torch.cumsum(p[:-1], dim=0)

                    step = i * w + j
                    shift = self.precision * (N - 1 - step)
                    target = V >> shift

                    sym = 0
                    for v in range(len(cdf)):
                        if int(cdf[v]) <= target < int(cdf[v] + p[v]):
                            sym = v
                            break

                    tokens[0, i, j] = sym
                    V = (V - (int(cdf[sym]) << shift)) // int(p[sym])

            return self.tokenizer.decode_index(tokens)[0]


def compress(tokenizer: Path, autoregressive: Path, image: Path, compressed_image: Path):
    """
    Compress images using a pre-trained model.

    tokenizer: Path to the tokenizer model.
    autoregressive: Path to the autoregressive model.
    images: Path to the image to compress.
    compressed_image: Path to save the compressed image tensor.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))
    cmp = Compressor(tk_model, ar_model)

    x = torch.tensor(np.array(Image.open(image)), dtype=torch.uint8, device=device)
    cmp_img = cmp.compress(x.float() / 255.0 - 0.5)
    with open(compressed_image, "wb") as f:
        f.write(cmp_img)


def decompress(tokenizer: Path, autoregressive: Path, compressed_image: Path, image: Path):
    """
    Decompress images using a pre-trained model.

    tokenizer: Path to the tokenizer model.
    autoregressive: Path to the autoregressive model.
    compressed_image: Path to the compressed image tensor.
    images: Path to save the image to compress.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))
    cmp = Compressor(tk_model, ar_model)

    with open(compressed_image, "rb") as f:
        cmp_img = f.read()

    x = cmp.decompress(cmp_img)
    img = Image.fromarray(((x + 0.5) * 255.0).clamp(min=0, max=255).byte().cpu().numpy())
    img.save(image)


if __name__ == "__main__":
    from fire import Fire

    Fire({"compress": compress, "decompress": decompress})
