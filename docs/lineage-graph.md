# Tag lineage graphs

Renderable on GitHub (native mermaid). The first graph is generated; do not
hand-edit it:

    scripts/lineage.sh graph --mermaid

(needs `IC_DIR` / `IL_DIR` mirrors, see docs/lineage.md). A static SVG can be
produced from the same source with mermaid-cli if one is ever required.

## Top-level pin lineage

Product order over InfiniCore x InfiniLM, Oct-Nov scope: 22 pins across four
streams. Dashed edge = declared `BASE` (orphan snapshot seed), which ancestry
cannot derive. Node label suffix `= ...` marks a republished tag (identical pin
vector).

```mermaid
flowchart TD
  subgraph SG_qy["qy"]
    direction TB
    Nv20261009qyminicpm516a3postrefactor["v2026.10.09-qy-minicpm5-16a3-post-refactor"]
    Nv20261010qyminicpm516a3fa["v2026.10.10-qy-minicpm5-16a3-fa<br/>= v2026.10.10-qy-minicpm5-2b-fa"]
    Nv20261010qyminicpm5cudagraph["v2026.10.10-qy-minicpm5-cudagraph"]
    Nv20261013qyminicpm516a3falongctxcudagraph["v2026.10.13-qy-minicpm5-16a3-fa-longctx-cudagraph"]
    Nv20261019qyminicpm5fullpiecewisecudagraph["v2026.10.19-qy-minicpm5-full-piecewise-cudagraph"]
    Nv20261021qyminicpm52bfullpiecewise["v2026.10.21-qy-minicpm5-2b-full-piecewise<br/>= v2026.10.22-qy-minicpm5-2b-full-piecewise"]
    Nv20261023qyminicpm52bbatchsamplefix["v2026.10.23-qy-minicpm5-2b-batch-sample-fix"]
    Nv20261023qyminicpm52bfullpiecewise["v2026.10.23-qy-minicpm5-2b-full-piecewise"]
    Nv20261027qyminicpm52bfacontig["v2026.10.27-qy-minicpm5-2b-fa-contig"]
    Nv20261027qyminicpm52bfullpiecewise["v2026.10.27-qy-minicpm5-2b-full-piecewise"]
    Nv20261028qykernelempending["v2026.10.28-qy-kernel-em-pending"]
    Nv20261029infinilmprodqwen38tp4["v2026.10.29-infinilm-prod-qwen38-tp4"]
    Nv20261112infinilmprodqwen38tp4arsplit["v2026.11.12-infinilm-prod-qwen38-tp4-ar-split"]
    Nv20261116infinilmprodqwen38tp4arsplit["v2026.11.16-infinilm-prod-qwen38-tp4-ar-split"]
    Nv20261122infinilmprodqwen38tp4marsrepro["v2026.11.22-infinilm-prod-qwen38-tp4-mars-repro"]
    Nv20261124infinilmshowroomeagercatchup["v2026.11.24-infinilm-showroom-eager-catchup<br/>= v2026.11.25-infinilm-showroom-freeze"]
  end
  subgraph SG_16a3["16a3"]
    direction TB
    Nv20261012qyminicpm516a3["v2026.10.12-qy-minicpm5-16a3"]
  end
  subgraph SG_metax["metax"]
    direction TB
    Nv20261014metaxg9v339a5b["v2026.10.14-metax-g9v3-39a5b"]
  end
  subgraph SG_qymoe["qy-moe"]
    direction TB
    Nv20261103qylmheadtpnccl["v2026.11.03-qy-lmhead-tp-nccl"]
    Nv20261106qyfullpd["v2026.11.06-qy-full-pd"]
    Nv20261109qymoebf16warpnk["v2026.11.09-qy-moe-bf16-warpnk<br/>= v2026.11.12-qy-moe-bf16-warpnk"]
    Nv20261109qymoedecodefull["v2026.11.09-qy-moe-decode-full"]
  end
  Nv20261009qyminicpm516a3postrefactor --> Nv20261010qyminicpm516a3fa
  Nv20261010qyminicpm516a3fa --> Nv20261010qyminicpm5cudagraph
  Nv20261010qyminicpm5cudagraph --> Nv20261013qyminicpm516a3falongctxcudagraph
  Nv20261013qyminicpm516a3falongctxcudagraph --> Nv20261019qyminicpm5fullpiecewisecudagraph
  Nv20261019qyminicpm5fullpiecewisecudagraph --> Nv20261021qyminicpm52bfullpiecewise
  Nv20261023qyminicpm52bfullpiecewise --> Nv20261023qyminicpm52bbatchsamplefix
  Nv20261021qyminicpm52bfullpiecewise --> Nv20261023qyminicpm52bfullpiecewise
  Nv20261027qyminicpm52bfullpiecewise --> Nv20261027qyminicpm52bfacontig
  Nv20261023qyminicpm52bfullpiecewise --> Nv20261027qyminicpm52bfullpiecewise
  Nv20261023qyminicpm52bbatchsamplefix --> Nv20261028qykernelempending
  Nv20261027qyminicpm52bfacontig --> Nv20261029infinilmprodqwen38tp4
  Nv20261028qykernelempending -. "declared BASE" .-> Nv20261103qylmheadtpnccl
  Nv20261103qylmheadtpnccl --> Nv20261106qyfullpd
  Nv20261109qymoedecodefull --> Nv20261109qymoebf16warpnk
  Nv20261106qyfullpd --> Nv20261109qymoedecodefull
  Nv20261029infinilmprodqwen38tp4 --> Nv20261112infinilmprodqwen38tp4arsplit
  Nv20261112infinilmprodqwen38tp4arsplit --> Nv20261116infinilmprodqwen38tp4arsplit
  Nv20261116infinilmprodqwen38tp4arsplit --> Nv20261122infinilmprodqwen38tp4marsrepro
  Nv20261122infinilmprodqwen38tp4marsrepro --> Nv20261124infinilmshowroomeagercatchup
```

Note: `v2026.10.28-qy-kernel-em-pending` is an InfiniLM ancestor of
`v2026.11.24` but not an InfiniCore ancestor, so it is not in that tag's
lineage - the product order, not either leaf alone.

## Nested dependency leaves of the refreeze pin

```mermaid
flowchart LR
  subgraph REF["InfiniCore pin f9c11542 - v2026.11.25-infinilm-showroom-freeze"]
    IC["InfiniCore f9c11542"]
    CCL["submodules/InfiniCCL<br/>13362553"]
    OPS["submodules/InfiniOps<br/>da91f734"]
    RT["submodules/InfiniRT<br/>63ad214f"]
    IC --> CCL
    IC --> OPS
    IC --> RT
  end
  MCCL["InfiniCCL master"]
  MOPS["InfiniOps master"]
  MRT["InfiniRT master"]
  CCL -. "shared with master through 52a48aac" .-> MCCL
  OPS -. "shared with master through 8c2f70a2" .-> MOPS
  RT -. "shared with master through 6b256e0a" .-> MRT
```

InfiniOps and InfiniRT share with their masters only through their
v2026.10.09-era pins - the same picture as InfiniCore itself.

## InfiniCCL divergence

```mermaid
flowchart TD
  C21["InfiniCCL c21a5b46<br/>fork point; also the 16a3 / metax CCL pin"]
  QY["8dda36c2<br/>qy trunk CCL pin<br/>(v2026.10.13 .. v2026.11.12-prod)"]
  PROD["13362553<br/>prod CCL pin<br/>(v2026.11.16 .. v2026.11.25)"]
  C21 --> QY
  C21 --> PROD
```

`13362553` does not contain `8dda36c2`. A nested-aware product order would
break the published parent chain between `v2026.11.12-prod` and
`v2026.11.16` - recorded but not yet applied (docs/lineage.md section 3.6).

## qy-moe orphan pins

```mermaid
flowchart LR
  QM["qy-moe snapshot pins<br/>InfiniCCL 960ecca6 / InfiniOps 194e7055 / InfiniRT 229c6691"]
  UP["current module refs"]
  QM -. "no shared history with master; upstream rejects as unreachable refs" .-> UP
```
