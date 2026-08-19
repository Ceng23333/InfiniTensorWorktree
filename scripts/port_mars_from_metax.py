#!/usr/bin/env python3
"""Copy MetaX/MACA device files to Mars/HPCC. Run from the worktree root."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "InfiniCore/submodules/InfiniOps"
CCL = ROOT / "InfiniCore/submodules/InfiniCCL"


def replace_ops(text: str) -> str:
    pairs = [
        ("INFINI_OPS_METAX_GEMM_MCBLAS_H_", "INFINI_OPS_MARS_GEMM_HCBLAS_H_"),
        ("INFINI_OPS_METAX", "INFINI_OPS_MARS"),
        ("native/cuda/metax", "native/cuda/mars"),
        ("infini/rt/metax", "infini/rt/mars"),
        ("Device::Type::kMetax", "Device::Type::kMars"),
        ("<mcblas/mcblas.h>", "<hcblas/hcblas.h>"),
        ("mcblasHandle_t", "hcblasHandle_t"),
        ("MCBLAS_OP_N", "HCBLAS_OP_N"),
        ("MCBLAS_OP_T", "HCBLAS_OP_T"),
        ("MACA_R_16F", "HPCC_R_16F"),
        ("MACA_R_16BF", "HPCC_R_16BF"),
        ("MACA_R_32F", "HPCC_R_32F"),
        ("MCBLAS_COMPUTE_32F_FAST_TF32", "HCBLAS_COMPUTE_32F_FAST_TF32"),
        ("MCBLAS_COMPUTE_32F", "HCBLAS_COMPUTE_32F"),
        ("MCBLAS_GEMM_DEFAULT", "HCBLAS_GEMM_DEFAULT"),
        ("mcblasGemmStridedBatchedEx", "hcblasGemmStridedBatchedEx"),
        ("mcblasSetStream", "hcblasSetStream"),
        ("mcblasCreate", "hcblasCreate"),
        ("mcblasDestroy", "hcblasDestroy"),
        ("__maca_bfloat16", "__hpcc_bfloat16"),
        ("MCR device properties query for Metax", "HCR device properties query for Mars"),
        ("TODO: Add MCR", "TODO: Add HCR"),
    ]
    for old, new in pairs:
        text = text.replace(old, new)
    return text


def replace_ccl_device(text: str) -> str:
    pairs = [
        ("INFINI_CCL_DEVICES_METAX", "INFINI_CCL_DEVICES_MARS"),
        ("devices/metax", "devices/mars"),
        ("Device::Type::kMetax", "Device::Type::kMars"),
        ("<mcr/mc_runtime.h>", "<hcr/hc_runtime.h>"),
        ("<common/maca_bfloat16.h>", "<common/hpcc_bfloat16.h>"),
        ("<common/maca_fp16.h>", "<common/hpcc_fp16.h>"),
        ("__maca_bfloat16", "__hpcc_bfloat16"),
        ("mcPointerAttribute_t", "hcPointerAttribute_t"),
        ("INFINI_CHECK_MACA", "INFINI_CHECK_HPCC"),
        ("mcPointerGetAttributes", "hcPointerGetAttributes"),
        ("mcMemoryTypeDevice", "hcMemoryTypeDevice"),
        ("mcMemoryTypeManaged", "hcMemoryTypeManaged"),
        ("mcMemoryTypeArray", "hcMemoryTypeArray"),
        ("CheckMacaImpl", "CheckHpccImpl"),
        ("mcError_t", "hcError_t"),
        ("maca_result", "hpcc_result"),
        ("mcSuccess", "hcSuccess"),
        ("mcGetLastError", "hcGetLastError"),
        ("MACA error code", "HPCC error code"),
        ("mcStream_t", "hcStream_t"),
        ("mcGetErrorString", "hcGetErrorString"),
        ("mcMalloc", "hcMalloc"),
        ("mcMemcpyHostToDevice", "hcMemcpyHostToDevice"),
        ("mcMemcpyDeviceToHost", "hcMemcpyDeviceToHost"),
        ("mcMemcpy", "hcMemcpy"),
        ("mcFree", "hcFree"),
        ("mcMemset", "hcMemset"),
        ("mcGetDevice", "hcGetDevice"),
        ("mcSetDevice", "hcSetDevice"),
        ("mcDeviceSynchronize", "hcDeviceSynchronize"),
        ("mcStreamSynchronize", "hcStreamSynchronize"),
    ]
    for old, new in pairs:
        text = text.replace(old, new)
    return text


def replace_hccl(text: str) -> str:
    pairs = [
        ("INFINI_CCL_BACKENDS_CCL_MCCL", "INFINI_CCL_BACKENDS_CCL_HCCL"),
        ("backends/ccl/mccl", "backends/ccl/hccl"),
        ("devices/metax", "devices/mars"),
        ("Device::Type::kMetax", "Device::Type::kMars"),
        ("BackendType::kMccl", "BackendType::kHccl"),
        ("McclDataTypeTraits", "HcclDataTypeTraits"),
        ("McclDataTypeMap", "HcclDataTypeMap"),
        ("McclApi", "HcclApi"),
        ("DataTypeToMcclType", "DataTypeToHcclType"),
        ("RedOpToMcclOp", "RedOpToHcclOp"),
        ("kMcclOpMap", "kHcclOpMap"),
        ("INFINI_CHECK_MCCL", "INFINI_CHECK_HCCL"),
        ("CheckMcclImpl", "CheckHcclImpl"),
        ("mccl_result", "hccl_result"),
        ("mccl_dtype", "hccl_dtype"),
        ("<mccl.h>", "<hccl.h>"),
        ("mcclBfloat16", "hcclBfloat16"),
        ("mcclFloat16", "hcclFloat16"),
        ("mcclFloat32", "hcclFloat32"),
        ("mcclFloat64", "hcclFloat64"),
        ("mcclInt8", "hcclInt8"),
        ("mcclInt32", "hcclInt32"),
        ("mcclInt64", "hcclInt64"),
        ("mcclUint8", "hcclUint8"),
        ("mcclUint32", "hcclUint32"),
        ("mcclUint64", "hcclUint64"),
        ("mcclNumTypes", "hcclNumTypes"),
        ("mcclDataType_t", "hcclDataType_t"),
        ("mcclRedOp_t", "hcclRedOp_t"),
        ("mcclComm_t", "hcclComm_t"),
        ("mcclUniqueId", "hcclUniqueId"),
        ("mcclResult_t", "hcclResult_t"),
        ("mcclGetErrorString", "hcclGetErrorString"),
        ("mcclGetUniqueId", "hcclGetUniqueId"),
        ("mcclCommInitRank", "hcclCommInitRank"),
        ("mcclCommDestroy", "hcclCommDestroy"),
        ("mcclAllReduce", "hcclAllReduce"),
        ("mcclSuccess", "hcclSuccess"),
        ("mcclSum", "hcclSum"),
        ("mcclProd", "hcclProd"),
        ("mcclMax", "hcclMax"),
        ("mcclMin", "hcclMin"),
        ("mcclAvg", "hcclAvg"),
        ("backend(mccl) MCCL", "backend(hccl) HCCL"),
        ("MCCL backend", "HCCL backend"),
        ("MCCL", "HCCL"),
        ("mccl", "hccl"),
    ]
    for old, new in pairs:
        text = text.replace(old, new)
    return text


def copy_tree(src: Path, dst: Path, transform, rename=None):
    if dst.exists():
        shutil.rmtree(dst)
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        out_rel = Path(rename(rel)) if rename else rel
        out = dst / out_rel
        out.parent.mkdir(parents=True, exist_ok=True)
        text = transform(path.read_text(encoding="utf-8"))
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out.relative_to(ROOT)}")


def ops_rename(rel: Path) -> Path:
    parts = list(rel.parts)
    if parts[-1] == "mcblas.h":
        parts[-1] = "hcblas.h"
    return Path(*parts)


def main() -> None:
    copy_tree(
        OPS / "src/native/cuda/metax",
        OPS / "src/native/cuda/mars",
        replace_ops,
        ops_rename,
    )
    copy_tree(
        CCL / "src/devices/metax",
        CCL / "src/devices/mars",
        replace_ccl_device,
    )

    src_mccl = CCL / "src/backends/ccl/mccl"
    dst_hccl = CCL / "src/backends/ccl/hccl"
    if dst_hccl.exists():
        shutil.rmtree(dst_hccl)
    for path in src_mccl.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src_mccl)
        if rel.parts and rel.parts[0] == "moore":
            continue
        if rel.parts[:1] == ("metax",):
            rel = Path("mars") / Path(*rel.parts[1:])
        out = dst_hccl / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(replace_hccl(path.read_text(encoding="utf-8")), encoding="utf-8")
        print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
