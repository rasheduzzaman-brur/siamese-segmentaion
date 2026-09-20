"""PyTorch -> ONNX -> OpenVINO IR export.

    python -m export.export_openvino --config configs/config.yaml --weights runs/.../best.pt \
        --out export/artifacts/siamese_defect

Design choice (see DESIGN.md section 15): NMS and mask assembly are picked
apart from the exported graph. Fixed input size + a per-point dense output
(boxes, scores, mask coefficients, prototypes) exports cleanly to ONNX/IR
with zero custom ops; NMS + mask crop/resize run as a small, fast NumPy/CPU
post-processing step in the application layer (identical math to
models.siamese_instance_segmentation.postprocess, just outside the traced
graph). This sidesteps the two main OpenVINO export pain points:
  1. torchvision.ops.batched_nms has a dynamic-size output that some IR
     versions handle poorly across OpenVINO releases.
  2. affine_grid/grid_sample based RoI cropping is supported by ONNX
     opset>=16 but is slower to validate operator-by-operator than doing
     the (tiny, per-detection) crop on CPU after NMS has already reduced
     the instance count to a handful.
"""
from __future__ import annotations

import argparse
from typing import Dict, List

import torch
import torch.nn as nn
import yaml

from models.siamese_instance_segmentation import SiameseInstanceSegmentation


class ExportWrapper(nn.Module):
    """Flattens the nested per-level dict output into fixed-shape tensors
    concatenated across all FPN levels -- required because ONNX tracing
    does not support returning a dict-of-dicts with a variable number of
    keys cleanly across opset/export-tool versions."""

    def __init__(self, model: SiameseInstanceSegmentation):
        super().__init__()
        self.model = model
        self.levels = ("p3", "p4", "p5", "p6", "p7")

    def forward(self, reference: torch.Tensor, inspected: torch.Tensor):
        out = self.model(reference, inspected)
        cls_list, box_list, ctr_list, coeff_list, point_list = [], [], [], [], []

        for lvl in self.levels:
            pred = out["levels"][lvl]
            b, c, h, w = pred["cls_logits"].shape
            cls_list.append(pred["cls_logits"].permute(0, 2, 3, 1).reshape(b, h * w, c))
            box_list.append(pred["box_reg"].permute(0, 2, 3, 1).reshape(b, h * w, 4))
            ctr_list.append(pred["centerness"].permute(0, 2, 3, 1).reshape(b, h * w, 1))
            k = pred["mask_coeff"].shape[1]
            coeff_list.append(pred["mask_coeff"].permute(0, 2, 3, 1).reshape(b, h * w, k))
            point_list.append(out["points"][lvl].unsqueeze(0).expand(b, -1, -1))

        return (
            torch.cat(cls_list, dim=1),      # (B, N_total, num_classes)
            torch.cat(box_list, dim=1),        # (B, N_total, 4)
            torch.cat(ctr_list, dim=1),         # (B, N_total, 1)
            torch.cat(coeff_list, dim=1),        # (B, N_total, K)
            torch.cat(point_list, dim=1),         # (B, N_total, 2)
            out["prototypes"],                     # (B, K, Hp, Wp)
            out["reference_embedding"],               # (B, D)
            out["inspected_embedding"],                 # (B, D)
        )


def export_onnx(cfg: dict, weights_path: str, onnx_path: str) -> None:
    device = torch.device("cpu")  # export on CPU for a deployment-representative graph
    model = SiameseInstanceSegmentation(cfg).to(device)
    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt.get("ema_model", ckpt.get("model", ckpt)))
    model.eval()

    wrapper = ExportWrapper(model)
    h, w = cfg["data"]["image_size"]
    dummy_ref = torch.randn(1, 3, h, w)
    dummy_inp = torch.randn(1, 3, h, w)

    torch.onnx.export(
        wrapper, (dummy_ref, dummy_inp), onnx_path,
        input_names=["reference", "input"],
        output_names=["cls_logits", "box_reg", "centerness", "mask_coeff",
                       "points", "prototypes",
                       "reference_embedding", "inspected_embedding"],
        opset_version=cfg["export"]["onnx_opset"],
        dynamic_axes=None if not cfg["export"]["dynamic_axes"] else {
            "reference": {0: "batch"}, "input": {0: "batch"},
        },
        do_constant_folding=True,
    )
    print(f"Exported ONNX graph to {onnx_path}")


def convert_to_openvino(onnx_path: str, ir_dir: str, precision: str = "FP16") -> None:
    """Requires `openvino-dev` (provides the `ovc`/`mo` Python API)."""
    import openvino as ov

    core = ov.Core()
    ov_model = ov.convert_model(onnx_path)
    ov.save_model(ov_model, f"{ir_dir}.xml", compress_to_fp16=(precision == "FP16"))
    print(f"Saved OpenVINO IR to {ir_dir}.xml / {ir_dir}.bin (precision={precision})")


def quantize_int8(ir_xml_path: str, calibration_loader, out_xml_path: str) -> None:
    """Post-training INT8 quantization via NNCF, using a small representative
    calibration set (a few hundred real reference/input pairs is typically
    enough -- see DESIGN.md section 15 for the accuracy-validation protocol
    that must follow any INT8 conversion before it is trusted on the line).
    """
    import nncf
    import openvino as ov

    core = ov.Core()
    model = core.read_model(ir_xml_path)

    def transform_fn(data_item):
        reference, inspected = data_item
        return {"reference": reference.numpy(), "input": inspected.numpy()}

    calibration_dataset = nncf.Dataset(calibration_loader, transform_fn)
    quantized_model = nncf.quantize(model, calibration_dataset, subset_size=300)
    ov.save_model(quantized_model, out_xml_path)
    print(f"Saved INT8-quantized OpenVINO IR to {out_xml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", required=True, help="output path prefix, no extension")
    parser.add_argument("--precision", default="FP16", choices=["FP32", "FP16", "INT8"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    onnx_path = args.out + ".onnx"
    export_onnx(cfg, args.weights, onnx_path)
    convert_to_openvino(onnx_path, args.out, precision="FP16" if args.precision != "FP32" else "FP32")

    if args.precision == "INT8":
        print("INT8 requested: run quantize_int8() separately with a real "
              "calibration DataLoader of (reference, input) tensor pairs.")
