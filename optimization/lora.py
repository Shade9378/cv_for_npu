"""
lora.py
-------
LoRA adapter injection, training, and weight merging for YOLOv8.

Usage:
    python lora.py --data /path/to/data.yaml [options]
"""

import math
import argparse
import copy
import torch
import torch.nn as nn
from ultralytics import YOLO

class LoRAConv2d(nn.Module):
    def __init__(self, base_conv: nn.Conv2d, rank: int = 4, alpha: float = 8.0):
        super().__init__()
        if not isinstance(base_conv, nn.Conv2d):
            raise TypeError("LoRAConv2d expects an nn.Conv2d module.")

        self.base  = base_conv
        self.rank  = rank
        self.alpha = alpha
        self.scale = alpha / rank

        for p in self.base.parameters():
            p.requires_grad = False

        in_ch  = base_conv.in_channels
        out_ch = base_conv.out_channels

        if base_conv.groups != 1:
            raise NotImplementedError("LoRAConv2d assumes groups=1.")

        self.lora_down = nn.Conv2d(in_ch,  rank,   kernel_size=1, bias=False)
        self.lora_up   = nn.Conv2d(rank,   out_ch, kernel_size=1, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.base(x) + self.scale * self.lora_up(self.lora_down(x))

    def merge(self) -> nn.Conv2d:
        """
            Merge LoRA weights back into the base conv and return a plain Conv2d.
            The merged weight is: W_merged = W_base + scale * (W_up @ W_down)
            Both W_up and W_down are 1x1 convs so this is a simple matmul.
        """
        merged_conv = copy.deepcopy(self.base)
        merged_conv.weight.requires_grad_(True)

        W_up = self.lora_up.weight.data.squeeze(-1).squeeze(-1)
        W_down = self.lora_down.weight.data.squeeze(-1).squeeze(-1)
        delta = W_up @ W_down

        delta_kernel = torch.zeros_like(merged_conv.weight.data)
        kH, kW = self.base.kernel_size
        cy, cx = kH // 2, kW // 2
        delta_kernel[:, :, cy, cx] = delta

        merged_conv.weight.data = self.base.weight.data + self.scale * delta_kernel

        if merged_conv.bias is not None and self.base.bias is not None:
            merged_conv.bias.data = self.base.bias.data.clone()

        return merged_conv

def _default_predicate(name: str, module: nn.Module) -> bool:
    return (
        isinstance(module, nn.Conv2d)
        and module.kernel_size == (3, 3)
        and module.groups == 1
    )


def inject_lora(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 8.0,
    predicate=None,
) -> None:
    if predicate is None:
        predicate = _default_predicate

    for name, child in list(model.named_children()):
        if predicate(name, child):
            setattr(model, name, LoRAConv2d(child, rank=rank, alpha=alpha))
        else:
            inject_lora(child, rank=rank, alpha=alpha, predicate=predicate)


def mark_only_lora_trainable(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name


def merge_lora(model: nn.Module) -> None:
    for name, child in list(model.named_children()):
        if isinstance(child, LoRAConv2d):
            setattr(model, name, child.merge())
            print(f"[merge] {name}: LoRA merged into base conv")
        else:
            merge_lora(child)


def count_trainable(model: nn.Module) -> tuple[int, int]:
    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    return len(trainable), sum(x[1] for x in trainable)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLOv8 + LoRA training script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    model_g = parser.add_argument_group("Model")
    model_g.add_argument("--weights", default="yolov8/yolov8n.pt", help="Base YOLOv8 weights")

    lora_g = parser.add_argument_group("LoRA")
    lora_g.add_argument("--rank",  type=int,   default=8,    help="LoRA rank")
    lora_g.add_argument("--alpha", type=float, default=16.0, help="LoRA alpha scaling factor")

    data_g = parser.add_argument_group("Data")
    data_g.add_argument("--data",  required=True, help="Path to data.yaml")
    data_g.add_argument("--imgsz", type=int, default=640, help="Input image size")

    train_g = parser.add_argument_group("Training")
    train_g.add_argument("--epochs", type=int,   default=5,      help="Training epochs")
    train_g.add_argument("--batch",  type=int,   default=32,     help="Batch size")
    train_g.add_argument("--lr0",    type=float, default=0.0001, help="Initial learning rate")
    train_g.add_argument("--lrf",    type=float, default=None,   help="Final LR fraction (optional)")

    out_g = parser.add_argument_group("Output")
    out_g.add_argument("--project", default="cv_for_npu/yolov8/runs", help="Training output dir")
    out_g.add_argument("--export",  default=None, help="If set, merge LoRA and export ONNX to this path")

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    print(f"\nLoRA YOLOv8 Training")
    print(f"  Weights : {args.weights}")
    print(f"  LoRA    : rank={args.rank}, alpha={args.alpha}")
    print(f"  Data    : {args.data}")
    print(f"  Epochs  : {args.epochs}, batch={args.batch}, lr0={args.lr0}")
    print(f"  Project : {args.project}\n")

    yolo = YOLO(args.weights)
    net  = yolo.model

    inject_lora(net, rank=args.rank, alpha=args.alpha)
    mark_only_lora_trainable(net)

    n_tensors, n_params = count_trainable(net)
    print(f"[OK] LoRA injected — trainable tensors: {n_tensors}, params: {n_params:,}\n")

    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        pretrained=True,
        amp=False,
        project=args.project,
    )
    if args.lrf is not None:
        train_kwargs["lrf"] = args.lrf

    yolo.train(**train_kwargs)

    if args.export:
        print(f"\n[INFO] Merging LoRA weights before export...")
        merge_lora(net)
        print(f"[INFO] Exporting ONNX to: {args.export}")
        yolo.export(format="onnx", project=args.export)


if __name__ == "__main__":
    main()