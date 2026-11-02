#!/usr/bin/env python3
"""Compare matched InfiniLM FULL and vLLM PTI captures."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def kernel_total(summary, substring):
    rows=[]
    for name, st in summary["kernels"].items():
        if substring.lower() in name.lower(): rows.append(st)
    return {
      "sum_ms": sum(x["sum_ms"] for x in rows),
      "count": sum(x["count"] for x in rows),
      "mean_us": (sum(x["sum_ms"] for x in rows)*1000.0/sum(x["count"] for x in rows)) if sum(x["count"] for x in rows) else 0,
    }

def flash_proxy(summary):
    return kernel_total(summary, "J_flash_attn_kvcache_mha")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--vllm-summary", required=True, type=Path)
    ap.add_argument("--infinilm-summary", required=True, type=Path)
    ap.add_argument("--vllm-facts", required=True, type=Path)
    ap.add_argument("--infinilm-facts", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args=ap.parse_args()
    v=json.loads(args.vllm_summary.read_text()); i=json.loads(args.infinilm_summary.read_text())
    vf=json.loads(args.vllm_facts.read_text())["graph"]; inf=json.loads(args.infinilm_facts.read_text())["graph"]
    vp=flash_proxy(v); ip=flash_proxy(i)
    def norm(summary, total_ms, proxy): return total_ms*1000.0/proxy["count"] if proxy["count"] else 0
    categories={
      "flash_attention": (kernel_total(v,"J_flash_attn_kvcache_mha"), kernel_total(i,"J_flash_attn_kvcache_mha")),
      "residual_norm": (kernel_total(v,"fused_add_rms_norm"), {**kernel_total(i,"ops9AddKernel"), **{}}),
    }
    # Build explicit normalized categories.
    rows=[]
    def add(label,vk,ik):
        vpms=vk["sum_ms"]*1000/vp["count"] if vp["count"] else 0
        ipms=ik["sum_ms"]*1000/ip["count"] if ip["count"] else 0
        rows.append({"bucket":label,"vllm_sum_ms":vk["sum_ms"],"infinilm_sum_ms":ik["sum_ms"],
                     "vllm_count":vk["count"],"infinilm_count":ik["count"],
                     "vllm_us_per_flash":vpms,"infinilm_us_per_flash":ipms,
                     "infinilm_over_vllm":ipms/vpms if vpms else None})
    add("flash_attention", kernel_total(v,"J_flash_attn_kvcache_mha"), kernel_total(i,"J_flash_atnkvcache_mha") if False else kernel_total(i,"J_flash_attn_kvcache_mha"))
    add("residual_add_rmsnorm", kernel_total(v,"fused_add_rms_norm"),
        {"sum_ms":kernel_total(i,"ops9AddKernel")["sum_ms"]+kernel_total(i,"RmsNormKernel")["sum_ms"],
         "count":kernel_total(i,"ops9AddKernel")["count"]+kernel_total(i,"RmsNormKernel")["count"]})
    add("rope", kernel_total(v,"rotary_embedding_kernel"), kernel_total(i,"ropeThreadPerItemKernel"))
    add("silu_act", kernel_total(v,"act_and_mul_kernel"), kernel_total(i,"SiluAndMulKernel"))
    add("kv_cache_reshape", kernel_total(v,"reshape_and_cache_flash_kernel"), kernel_total(i,"ReshapeAndCacheFlashKernel"))
    vcu=v["kind_sum_ms"].get("CU",0); icu=i["kind_sum_ms"].get("CU",0)
    vtu=v["kind_sum_ms"].get("TU",0); itu=i["kind_sum_ms"].get("TU",0)
    vm=v["kind_sum_ms"].get("MEMCPY",0); im=i["kind_sum_ms"].get("MEMCPY",0)
    total={"vllm":{"cu_ms":vcu,"tu_ms":vtu,"memcpy_ms":vm,"cu_per_flash_us":vcu*1000/vp["count"],"tu_per_flash_us":vtu*1000/vp["count"],"memcpy_per_flash_us":vm*1000/vp["count"]},
           "infinilm":{"cu_ms":icu,"tu_ms":itu,"memcpy_ms":im,"cu_per_flash_us":icu*1000/ip["count"],"tu_per_flash_us":itu*1000/ip["count"],"memcpy_per_flash_us":im*1000/ip["count"]}}
    total["infinilm_over_vllm"]={
      "cu_per_flash":total["infinilm"]["cu_per_flash_us"]/total["vllm"]["cu_per_flash_us"],
      "tu_per_flash":total["infinilm"]["tu_per_flash_us"]/total["vllm"]["tu_per_flash_us"],
      "memcpy_per_flash":total["infinilm"]["memcpy_per_flash_us"]/total["vllm"]["memcpy_per_flash_us"]}
    out={"normalization_proxy":{"vllm_flash_calls":vp["count"],"infinilm_flash_calls":ip["count"]},
         "device_totals":total,"categories":rows,
         "graph":{"vllm":vf,"infinilm":inf}}
    args.output.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out,indent=2))
if __name__=="__main__": main()
