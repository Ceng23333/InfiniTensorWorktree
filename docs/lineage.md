# Lineage for published tags

Status: proposal. Scope: `InfiniTensorWorktree` (ITW) and its leaf repos
(`InfiniCore`, `InfiniLM`). Audience: release owners and anyone who has to answer
"what does this release descend from".

## 1. Problem

A published tag freezes a *vector* of commits: one gitlink per leaf plus a
`MANIFEST`. Nothing in the repo records how that vector relates to the previous
one. Today the only lineage signals are the tag name and commit ancestry in each
leaf, and neither is queryable:

- Tags are immutable, so lineage cannot be attached to them after publication.
- Leaf branches move. A pinned commit stops being reachable from any branch head,
  so ancestry cannot be recomputed from a fresh clone.
- Some streams are seeded by orphan commits whose only link to their parent is a
  commit *subject* (`pin <sha>`). Git reports no relationship at all.

The result is that lineage is tribal knowledge, and every consumer re-derives it
by hand or trusts a date in a tag name.

## 2. Evidence from the Oct-Nov set

25 pinned tags, 22 distinct pin vectors, 4 independent sub-lineages. The 26th
Oct-Nov tag, `v2026.11.04-ar-cut-locked`, has no pin vector and is excluded.

| Stream   | Pins | Tag dates     | History                                          |
|----------|------|---------------|--------------------------------------------------|
| `qy`     | 16   | 10.09 - 11.24 | shared trunk, 662 commits deep                   |
| `qy-moe` | 4    | 11.03 - 11.12 | orphan, 2 commits deep, seeded from `v2026.10.28` |
| `16a3`   | 1    | 10.12         | fork of trunk at `270feb3e` (2026-09-11)         |
| `metax`  | 1    | 10.14         | fork of trunk at `270feb3e` (2026-09-11)         |

Four facts drive the design:

1. **Leaves move independently.** InfiniCore was pinned at `82dcc1f9` across ten
   tags while InfiniLM advanced on nearly every one. Lineage is therefore not a
   property of a tag date; it is a property of the pair.
2. **The sub-lineages agree across leaves.** Every tag lands in the same stream
   in both leaves, so one stream label per tag is sufficient.
3. **Publication is not pin identity.** Three tags republish an existing pin
   vector (`10.10-16a3-fa` = `10.10-2b-fa`, `10.21` = `10.22`,
   `11.09-qy-moe-bf16` = `11.12-qy-moe-bf16`). 25 pinned tags, 22 pins.
4. **Derived ancestry under-reports.** The `qy-moe` stream has no common ancestor
   with `qy` in either leaf. Derived alone, its four tags look unrelated to
   everything else, which is wrong: both orphan roots are `pin <sha>` commits
   naming `v2026.10.28-qy-kernel-em-pending`.

One more subtlety worth calling out, because it is the reason the model is a
product order and not a per-leaf merge: `v2026.10.28-qy-kernel-em-pending` is an
InfiniLM ancestor of `v2026.11.24-infinilm-showroom-eager-catchup`, but its
InfiniCore pin (`8c7dbde3`) is *not* an ancestor of `f9c11542`. It is therefore
not in the lineage of `v2026.11.24`, even though half of its pin vector is.

## 3. Model

### 3.1 Pin vector

For a published tag `T`, the pin vector is
`pi(T) = (pi_IC(T), pi_IL(T))`, read from the gitlinks in the tag's tree. Tags
with no gitlinks (currently `v2026.11.04-ar-cut-locked`) are outside the model and
must be listed in `lineage/UNPINNED`.

Two tags with the same pin vector are the same pin. Lineage is defined on pins;
a tag is a *publication* of a pin. This keeps republished tags from appearing as
their own ancestors.

### 3.2 Leaf ancestry

For leaf `l`, `a <=_l b` iff `pi_l(a)` is an ancestor of or equal to `pi_l(b)` in
`l`. This is the raw commit relation, computed per leaf with ordinary `git`.

### 3.3 Product-order lineage

`a <= b` iff `a <=_l b` for **every** leaf `l`. This is a partial order:
reflexive, antisymmetric on distinct pins, transitive. The lineage of `T` is
`L(T) = { U : U <= T }`.

The product order is strictly stronger than any per-leaf relation, and that is
the point: a release inherits only what it inherits in *all* leaves. The
`v2026.10.28` example above is the case that breaks per-leaf reasoning.

### 3.4 Sub-lineage

Within one leaf, the *comparability graph* `G_l` has an edge between two tags
whenever one is an ancestor of the other. A **sub-lineage** is a connected
component of `G_l`. Tags on disjoint histories (different orphan roots, or forks
that never merged back) land in different components.

Applied to the Oct-Nov set, this yields exactly four sub-lineages per leaf, and
the same four in both leaves. That agreement is what makes a single `STREAM`
label per tag legitimate rather than an assumption.

### 3.5 Declared edges

Two edge types, because derivation cannot see everything:

| Edge     | Meaning                                      | Source              |
|----------|----------------------------------------------|---------------------|
| `PARENT` | maximal strict ancestor in the product order | derived, verifiable |
| `BASE`   | publication whose pin seeded this history    | declared, provable  |

`BASE` exists for orphan histories. It is only legal when the orphan root commit
has subject `pin <sha>` and that sha, resolved in the same leaf, equals `BASE`'s
pin there. The sha is abbreviated in the wild (`pin 8c7dbde34c43`), so it is
resolved through `git rev-parse` rather than string-compared. The convention
already exists; this just makes it a contract instead of a commit message
convention.

### 3.6 Nested dependency leaves

InfiniCore is not a true leaf: it carries its own dependencies as git
submodules under `submodules/` (`InfiniCCL`, `InfiniOps`, `InfiniRT`), and
their pins move independently of the InfiniCore commit. They are discovered
from the pin's tree and recorded per tag as `PIN_IC_<MODULE>`, together with
`MAIN_BASE_IC_<MODULE>`: the newest commit that submodule shares with its own
default branch (`master`).

They are recorded, but they do not participate in the product order of section
3.3 yet. The Oct-Nov data shows why that is a separate decision and not a
formality: the `qy` trunk pins InfiniCCL at `8dda36c2`, while `v2026.11.16` and
later pin it at `13362553`, and `13362553` does not contain `8dda36c2` - the
two lines forked at `c21a5b46`. Folding nested leaves into the product order
would therefore disconnect `v2026.11.16+` from `v2026.11.12-prod` in the
published parent chain. Record first, decide second.

The `qy-moe` snapshot pins are the extreme case: its InfiniCCL / InfiniOps /
InfiniRT pins (`960ecca6`, `194e7055`, `229c6691`) share no history at all with
their modules' `master`, and the upstreams now reject them as unreachable refs.

Module repos resolve from `IC_DIR/submodules/<name>` (a recursive submodule
checkout) and can be pointed elsewhere with `IC_<MODULE>_DIR`; the nested
branch defaults to `master` and is overridable with `IC_<MODULE>_MAIN_REF`.

## 4. Storage

Lineage lives in the repo at `HEAD`, append-only. Tags are never rewritten.

### 4.1 `lineage/<tag>.env`

One file per published tag, dotenv style to match `MANIFEST`:

```dotenv
TAG=v2026.11.03-qy-lmhead-tp-nccl
PIN_IC=f11b043783991a5e6c8a30869de765b370f4da68
PIN_IL=6c34d02a479cac1ffec70f010d1d33ed5785a4f7
STREAM=qy-moe
KIND=snapshot
PARENT=
BASE=v2026.10.28-qy-kernel-em-pending
NOTE=orphan branch seeded from the 10.28 kernel checkpoint
```

| Field    | Required | Meaning                                                      |
|----------|----------|--------------------------------------------------------------|
| `TAG`    | yes      | must match the filename and `refs/tags`                      |
| `PIN_IC` | yes      | must equal the tag's InfiniCore gitlink                      |
| `PIN_IL` | yes      | must equal the tag's InfiniLM gitlink                        |
| `STREAM` | yes      | human label for the sub-lineage; not derivable from the tag name |
| `KIND`   | yes      | `pin`, `snapshot`, `fork`, `republish`                       |
| `PARENT` | yes      | comma-separated maximal ancestors; empty for a root          |
| `BASE`   | no       | declared seed; only legal on a root (`PARENT` empty)          |
| `MAIN_BASE_IC` | no | newest commit shared with the InfiniCore `main` branch        |
| `MAIN_BASE_IL` | no | newest commit shared with the InfiniLM `main` branch          |
| `PIN_IC_<MOD>` | yes | pin of InfiniCore's nested submodule `<MOD>`, must match the InfiniCore pin's tree |
| `MAIN_BASE_IC_<MOD>` | no | newest commit that submodule shares with its own `master` |
| `NOTE`   | no       | free text, one line                                          |

`KIND` values:

- `pin` - advances one or both leaves inside existing histories.
- `snapshot` - seeds a new (orphan) history from `BASE`.
- `fork` - first publication on a new branch of an existing history.
- `republish` - identical pin vector to an earlier tag.

`KIND` is computed inside the `TAG_GLOB` scope. A tag that is only a root
because earlier tags are out of scope shows as `fork`, so a partial migration
should be re-run over the full history before the labels are treated as final.

`STREAM` names the sub-lineage from section 3.4, not the deployment target. The
`infinilm-prod-*` tags sit on the `qy` trunk in both leaves and therefore carry
`STREAM=qy`; a prod tag that forked would be a different sub-lineage and get its
own name.

### 4.2 Tag annotation

New tags carry the lineage summary in the annotation, so `git show <tag>`
answers the question offline. Existing tags are left alone.

```text
release: freeze InfiniTensorWorktree v2026.11.24-infinilm-showroom-eager-catchup

IC_SHA=f9c11542a0dc77aa32445e827d78c729b2f63356
IL_SHA=08dc99ee3f8f995dae3b4b11be5ec8374ba7f097

Lineage: parent=v2026.11.22-infinilm-prod-qwen38-tp4-mars-repro
Stream:  qy
```

### 4.3 Generated index

`lineage.json` is generated from the env files and is the artifact consumers read
(`scripts/lineage.sh index`). It is committed, deterministic, and never edited by
hand.

## 5. Tooling

`scripts/lineage.sh`, pure bash + git:

| Command         | Output                                                          |
|-----------------|-----------------------------------------------------------------|
| `emit <tag>`    | print (or `--write`) the env file, deriving `PARENT` and `BASE`  |
| `emit --all`    | write one env file per tag in scope (`TAG_GLOB`)                |
| `graph`         | ASCII forest of the pin lineage, grouped by stream              |
| `explain <tag>` | per-leaf cone, product lineage, declared base                   |
| `verify`        | run the invariants in section 6                                 |
| `index`         | regenerate `lineage.json`                                       |

Leaf repos are resolved from `IC_DIR` / `IL_DIR`, defaulting to the sibling
submodules, so the tool works both in a full checkout and in a bare-mirror setup.
Scope is `TAG_GLOB` (default `v*`), so a partial migration can target
`TAG_GLOB='v2026.10.* v2026.11.*'`.
`MAIN_REF` names the branch that `MAIN_BASE_*` is computed against
(default `origin/main`).

## 6. Invariants

`verify` fails on any of these:

1. **Coverage.** Every tag in `refs/tags` has an env file or is listed in
   `lineage/UNPINNED`; every env file matches a tag.
2. **Pin fidelity.** `PIN_IC` / `PIN_IL` equal the tag's gitlinks.
3. **Parent validity.** Every `PARENT` is a strict product-order ancestor.
4. **Parent maximality.** No undeclared tag is a strict product-order ancestor
   that is not below some declared `PARENT`.
5. **Base legality.** `BASE` implies an orphan history; the orphan root subject
   is `pin <sha>`; that sha equals `BASE`'s pin in that leaf; both leaves name the
   same `BASE`.
6. **Acyclicity.** The `PARENT` / `BASE` graph is a DAG.
7. **Reachability.** Every pin is fetchable from the declared leaf URL. This
   catches the failure mode where a pin silently becomes unfetchable.
8. **Shared-with-main.** `MAIN_BASE_IC` / `MAIN_BASE_IL`, when present, must
   equal the `git merge-base` of that leaf's pin and `MAIN_REF`. Derived, not
   trusted.
9. **Nested pins.** `PIN_IC_<MOD>` must equal the gitlink at
   `submodules/<MOD>` of the tag's InfiniCore pin. `MAIN_BASE_IC_<MOD>`, when
   present, must equal that submodule's merge-base with its default branch,
   re-proved whenever the module repo is available.

## 7. Migration

1. Run `lineage.sh emit --all --write`; hand-review `STREAM` and `NOTE`.
2. `KIND=snapshot` and `BASE` for the `qy-moe` root; `KIND=republish` for the
   three duplicate pins; `KIND=fork` for `16a3` and `metax`.
3. `lineage/UNPINNED` gets `v2026.11.04-ar-cut-locked`.
4. Regenerate `lineage.json`, commit once. No tag is rewritten.
5. From the next release on, `release.sh` calls `lineage.sh emit "${TAG}" --write`
   before tagging and fails if `verify` is not clean.

## 8. Non-goals

- Not a package manager or a rebuild system. It records what a release was made
  of, not how to rebuild it.
- Does not replace `MANIFEST`, which stays the single-source pin record for
  consumers.
- Does not model semantic compatibility. A lineage edge means "contains the same
  commits", not "behaves the same".

## Appendix A - Oct-Nov lineage

`depth` is the length of the longest product-order chain below the pin.

| Pin (tags)                                                 | Stream | Parent                            | Depth |
|------------------------------------------------------------|--------|-----------------------------------|-------|
| `v2026.10.09-qy-minicpm5-16a3-post-refactor`               | qy     | -                                 | 0     |
| `v2026.10.10-qy-minicpm5-16a3-fa` = `...-2b-fa`            | qy     | `10.09-...-post-refactor`         | 1     |
| `v2026.10.10-qy-minicpm5-cudagraph`                        | qy     | `10.10-...-16a3-fa`               | 2     |
| `v2026.10.12-qy-minicpm5-16a3`                             | 16a3   | -                                 | 0     |
| `v2026.10.13-...-fa-longctx-cudagraph`                     | qy     | `10.10-...-cudagraph`             | 3     |
| `v2026.10.14-metax-g9v3-39a5b`                             | metax  | -                                 | 0     |
| `v2026.10.19-...-full-piecewise-cudagraph`                 | qy     | `10.13-...-longctx-cudagraph`     | 4     |
| `v2026.10.21-...-2b-full-piecewise` = `10.22-...`          | qy     | `10.19-...-piecewise-cudagraph`   | 5     |
| `v2026.10.23-...-2b-full-piecewise`                        | qy     | `10.21-...-2b-full-piecewise`     | 6     |
| `v2026.10.23-...-2b-batch-sample-fix`                      | qy     | `10.23-...-2b-full-piecewise`     | 7     |
| `v2026.10.27-...-2b-full-piecewise`                        | qy     | `10.23-...-2b-full-piecewise`     | 7     |
| `v2026.10.27-...-2b-fa-contig`                             | qy     | `10.27-...-2b-full-piecewise`     | 8     |
| `v2026.10.28-qy-kernel-em-pending`                         | qy     | `10.23-...-batch-sample-fix`      | 8     |
| `v2026.10.29-infinilm-prod-qwen38-tp4`                     | qy     | `10.27-...-fa-contig`             | 9     |
| `v2026.11.03-qy-lmhead-tp-nccl`                            | qy-moe | - (base `10.28-qy-kernel-em-pending`) | 0 |
| `v2026.11.06-qy-full-pd`                                   | qy-moe | `11.03-...-lmhead-tp-nccl`        | 1     |
| `v2026.11.09-qy-moe-decode-full`                           | qy-moe | `11.06-...-full-pd`               | 2     |
| `v2026.11.09-qy-moe-bf16-warpnk` = `11.12-...-bf16-warpnk` | qy-moe | `11.09-...-decode-full`           | 3     |
| `v2026.11.12-infinilm-prod-qwen38-tp4-ar-split`            | qy     | `10.29-...-tp4`                   | 10    |
| `v2026.11.16-...-tp4-ar-split`                             | qy     | `11.12-...-tp4-ar-split`          | 11    |
| `v2026.11.22-...-tp4-mars-repro`                           | qy     | `11.16-...-tp4-ar-split`          | 12    |
| `v2026.11.24-...-eager-catchup`                            | qy     | `11.22-...-mars-repro`            | 13    |

## Appendix B - the orphan claim

Both `qy-moe` histories are two commits deep and share no root with `qy`:

```text
InfiniCore  root 9c88f0b  2026-11-02  pin 8c7dbde34c43   <- IC pin of v2026.10.28
InfiniLM    root 29bf6a06 2026-11-02  pin b53db38ec20d   <- IL pin of v2026.10.28
```

`git merge-base` returns nothing for any `qy-moe` / `qy` pair in either leaf, so
no ancestry-based tool can recover the relationship. The `pin <sha>` subject is
the only evidence, which is why `BASE` is declared and then verified against it
rather than trusted.
