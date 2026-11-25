#!/usr/bin/env bash
# Derive and verify publication lineage for InfiniTensorWorktree tags.
#
# Lineage is defined on pin vectors, not on tags: a tag publishes a pin, and two
# tags with the same (IC, IL) pair are the same pin.
#
#   PARENT  maximal strict ancestor in the product order (ancestor in every leaf)
#   BASE    declared seed for an orphan history, proven by a `pin <sha>` root
#
# Usage:
#   scripts/lineage.sh graph
#   scripts/lineage.sh explain <tag>
#   scripts/lineage.sh emit <tag> [--write]
#   scripts/lineage.sh verify
#   scripts/lineage.sh index            # regenerate lineage.json
#
# Env:
#   IC_DIR / IL_DIR   leaf checkouts (default: sibling submodules)
#
# Leaf repos may be bare, but must not be shallow: ancestry beyond a shallow
# boundary is reported as unrelated, so run verify against a full mirror.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=worktree_env.sh
source "${SCRIPT_DIR}/worktree_env.sh"

ITW="${INFINI_TENSOR_WORKTREE}"
IC_DIR="${IC_DIR:-${ITW}/InfiniCore}"
IL_DIR="${IL_DIR:-${ITW}/InfiniLM}"
LINEAGE_DIR="${ITW}/lineage"
UNPINNED_FILE="${LINEAGE_DIR}/UNPINNED"
INDEX_FILE="${ITW}/lineage.json"
MAIN_REF="${MAIN_REF:-origin/main}"
TAG_GLOB="${TAG_GLOB:-v*}"
NESTED_PREFIX="submodules"   # InfiniCore's own dependency leaves

die() { echo "error: $*" >&2; exit 1; }

leaf_repo() {
  case "$1" in
    InfiniCore) echo "${IC_DIR}" ;;
    InfiniLM) echo "${IL_DIR}" ;;
    *) die "unknown leaf: $1" ;;
  esac
}

# Dependency leaves of InfiniCore live in its own tree under submodules/.
nested_field() { # <name> <prefix> -> e.g. MAIN_BASE_IC_INFINICCL
  echo "${2}IC_${1^^}"
}

nested_repo() { # <name> -> repo dir, or non-zero when unavailable
  local u="${1^^}"
  local v="IC_${u}_DIR"
  if [[ -n "${!v:-}" ]]; then echo "${!v}"; return 0; fi
  if [[ -d "${IC_DIR}/submodules/$1" ]]; then echo "${IC_DIR}/submodules/$1"; return 0; fi
  return 1
}

nested_main_ref() { # <name> -> branch to compute shared-with-main against
  local u="${1^^}"
  local v="IC_${u}_MAIN_REF"
  echo "${!v:-master}"
}

# ---------------------------------------------------------------- state

TAGS=()
declare -A PIN_IC=() PIN_IL=()      # tag -> pin sha
declare -A PIN_NESTED=()            # "tag|name" -> pin sha inside InfiniCore
NESTED_NAMES=()                     # dependency leaves discovered in InfiniCore
declare -A MASK_IC=() MASK_IL=()    # pin sha -> ancestor bitset, one char per pin
declare -A IDX_IC=() IDX_IL=()      # pin sha -> bit index
REPS=()                             # representative tag per distinct pin vector
declare -A DEPTH=()

OUT_PARENTS=""; OUT_BASE=""; OUT_DEPTH=0

env_field() { # <file> <field>
  local line out=""
  [[ -f "$1" ]] || { echo ""; return 0; }
  while IFS= read -r line; do
    line="${line%$'\r'}"
    if [[ "${line}" == "$2="* ]]; then out="${line#*=}"; break; fi
  done < "$1"
  echo "${out}"
}

pin_key() { echo "${PIN_IC[$1]}|${PIN_IL[$1]}"; }

same_pin() { [[ "${PIN_IC[$1]}|${PIN_IL[$1]}" == "${PIN_IC[$2]}|${PIN_IL[$2]}" ]]; }

is_tag() { local t; for t in "${TAGS[@]}"; do if [[ "${t}" == "$1" ]]; then return 0; fi; done; return 1; }

is_unpinned() {
  [[ -f "${UNPINNED_FILE}" ]] || return 1
  local t
  while IFS= read -r t; do
    t="${t%$'\r'}"
    if [[ "${t}" == "$1" ]]; then return 0; fi
  done < "${UNPINNED_FILE}"
  return 1
}

# ---------------------------------------------------------------- load

# Reachability over the union of pin histories, one git walk per leaf. Each
# commit carries a bitset of which pins it contains, so ancestry is O(1) after.
build_masks() { # <leaf> <repo> <sha...>
  local leaf="$1" repo="$2"
  shift 2
  local -a pins=("$@") order=()
  local -A par=() idx=() bits=()
  local i k n sha rest p pb nb ca cb
  n=${#pins[@]}
  for (( i = 0; i < n; i++ )); do idx["${pins[$i]}"]="${i}"; done

  while read -r sha rest; do
    order+=("${sha}")
    par["${sha}"]="${rest}"
  done < <(git -C "${repo}" rev-list --topo-order --parents "${pins[@]}" 2>/dev/null || true)

  for (( k = ${#order[@]} - 1; k >= 0; k-- )); do
    sha="${order[$k]}"
    nb=""
    for (( i = 0; i < n; i++ )); do
      if [[ "${idx[$sha]:-}" == "${i}" ]]; then nb+="1"; else nb+="0"; fi
    done
    for p in ${par[$sha]:-}; do
      pb="${bits[$p]:-}"
      [[ -z "${pb}" ]] && continue
      local merged=""
      for (( i = 0; i < n; i++ )); do
        ca="${nb:$i:1}"; cb="${pb:$i:1}"
        if [[ "${ca}" == "1" || "${cb}" == "1" ]]; then merged+="1"; else merged+="0"; fi
      done
      nb="${merged}"
    done
    bits["${sha}"]="${nb}"
  done

  for (( i = 0; i < n; i++ )); do
    sha="${pins[$i]}"
    if [[ "${leaf}" == InfiniCore ]]; then
      MASK_IC["${sha}"]="${bits[$sha]:-}"; IDX_IC["${sha}"]="${i}"
    else
      MASK_IL["${sha}"]="${bits[$sha]:-}"; IDX_IL["${sha}"]="${i}"
    fi
  done
}

existing() { # <repo> <sha...> -> shas that exist as commits
  local repo="$1"
  shift
  printf '%s\n' "$@" | git -C "${repo}" cat-file --batch-check 2>/dev/null |
    while read -r sha type _; do
      if [[ "${type}" == "commit" ]]; then echo "${sha}"; fi
    done
}

load() {
  local t out ic il line
  while IFS= read -r t; do
    [[ -z "${t}" ]] && continue
    out="$(git -C "${ITW}" ls-tree "${t}" InfiniCore InfiniLM 2>/dev/null || true)"
    ic=""; il=""
    while IFS= read -r line; do
      if [[ "${line}" =~ ^160000\ commit\ ([0-9a-f]{40})[[:space:]]+InfiniCore$ ]]; then
        ic="${BASH_REMATCH[1]}"
      fi
      if [[ "${line}" =~ ^160000\ commit\ ([0-9a-f]{40})[[:space:]]+InfiniLM$ ]]; then
        il="${BASH_REMATCH[1]}"
      fi
    done <<< "${out}"
    TAGS+=("${t}")
    PIN_IC["${t}"]="${ic}"
    PIN_IL["${t}"]="${il}"
  done < <(git -C "${ITW}" tag -l ${TAG_GLOB} | sort)

  # Tags whose pins are absent from the mirror cannot be placed.
  local -a ic_all=() il_all=() keep=() missing=()
  local -A ic_ok=() il_ok=()
  local sha
  for t in "${TAGS[@]}"; do
    ic="${PIN_IC[$t]}"; il="${PIN_IL[$t]}"
    [[ -n "${ic}" && -n "${il}" ]] || continue
    ic_all+=("${ic}"); il_all+=("${il}")
  done
  mapfile -t ic_all < <(printf '%s\n' "${ic_all[@]}" | sort -u)
  mapfile -t il_all < <(printf '%s\n' "${il_all[@]}" | sort -u)
  while read -r sha; do [[ -n "${sha}" ]] && ic_ok["${sha}"]=1; done < <(existing "$(leaf_repo InfiniCore)" "${ic_all[@]}")
  while read -r sha; do [[ -n "${sha}" ]] && il_ok["${sha}"]=1; done < <(existing "$(leaf_repo InfiniLM)" "${il_all[@]}")

  for t in "${TAGS[@]}"; do
    ic="${PIN_IC[$t]}"; il="${PIN_IL[$t]}"
    if [[ -n "${ic}" && -n "${il}" ]]; then
      if [[ -z "${ic_ok[$ic]:-}" || -z "${il_ok[$il]:-}" ]]; then
        missing+=("${t}")
        continue
      fi
    fi
    keep+=("${t}")
  done
  if [[ ${#missing[@]} -gt 0 ]]; then
    echo "warning: pins absent from the leaf mirrors, skipped: ${missing[*]}" >&2
  fi
  TAGS=("${keep[@]:-}")
  [[ ${#TAGS[@]} -gt 0 ]] || die "no pinned tags found"

  local -a ic_pins=() il_pins=()
  for sha in "${ic_all[@]}"; do [[ -n "${ic_ok[$sha]:-}" ]] && ic_pins+=("${sha}"); done
  for sha in "${il_all[@]}"; do [[ -n "${il_ok[$sha]:-}" ]] && il_pins+=("${sha}"); done
  build_masks InfiniCore "$(leaf_repo InfiniCore)" "${ic_pins[@]}"
  build_masks InfiniLM "$(leaf_repo InfiniLM)" "${il_pins[@]}"

  # Discover InfiniCore's dependency leaves and record their pins per tag.
  local -A nseen=()
  local nm line2
  for sha in "${ic_pins[@]}"; do
    while IFS= read -r line2; do
      if [[ "${line2}" =~ ^160000\ commit\ ([0-9a-f]{40})[[:space:]]+(.+)$ ]]; then
        nm="${BASH_REMATCH[2]}"; nm="${nm##*/}"
        if [[ -z "${nseen[${nm}]:-}" ]]; then nseen["${nm}"]=1; NESTED_NAMES+=("${nm}"); fi
      fi
    done < <(git -C "$(leaf_repo InfiniCore)" ls-tree "${sha}:${NESTED_PREFIX}" 2>/dev/null || true)
  done
  for t in "${TAGS[@]}"; do
    ic="${PIN_IC[${t}]}"
    for nm in "${NESTED_NAMES[@]:-}"; do
      PIN_NESTED["${t}|${nm}"]="$(git -C "$(leaf_repo InfiniCore)" rev-parse "${ic}:${NESTED_PREFIX}/${nm}" 2>/dev/null || true)"
    done
  done

  # Distinct pin vectors, ordered by first publication.
  local -A rep2=()
  local k
  REPS=()
  for t in "${TAGS[@]}"; do
    [[ -z "${PIN_IC[$t]:-}" ]] && continue
    k="${PIN_IC[$t]}|${PIN_IL[$t]}"
    if [[ -z "${rep2[$k]:-}" ]]; then rep2[$k]="${t}"; fi
  done
  mapfile -t REPS < <(printf '%s\n' "${rep2[@]}" | sort)
}

# ---------------------------------------------------------------- queries

is_anc() { # <leaf> <a> <b> -> 0 if a is ancestor-or-equal of b
  local leaf="$1" a="$2" b="$3" ia bits
  [[ "${a}" == "${b}" ]] && return 0
  if [[ "${leaf}" == InfiniCore ]]; then
    ia="${IDX_IC[$a]:-}"; bits="${MASK_IC[$b]:-}"
  else
    ia="${IDX_IL[$a]:-}"; bits="${MASK_IL[$b]:-}"
  fi
  [[ -n "${ia}" && -n "${bits}" ]] || return 1
  [[ "${bits:${ia}:1}" == "1" ]]
}

product_anc() { # <a> <b> -> 0 if a <= b in every leaf
  local a="$1" b="$2"
  [[ -n "${PIN_IC[$a]:-}" && -n "${PIN_IL[$a]:-}" ]] || return 1
  [[ -n "${PIN_IC[$b]:-}" && -n "${PIN_IL[$b]:-}" ]] || return 1
  if is_anc InfiniCore "${PIN_IC[$a]}" "${PIN_IC[$b]}"; then
    if is_anc InfiniLM "${PIN_IL[$a]}" "${PIN_IL[$b]}"; then return 0; fi
  fi
  return 1
}

pin_reps() { printf '%s\n' "${REPS[@]:-}"; }

max_parents() { # <tag> -> OUT_PARENTS (comma separated)
  local t="$1" u v skip
  local -a cand=() out=()
  OUT_PARENTS=""
  for u in "${REPS[@]}"; do
    if [[ "${u}" == "${t}" ]]; then continue; fi
    if same_pin "${u}" "${t}"; then continue; fi
    if product_anc "${u}" "${t}"; then cand+=("${u}"); fi
  done
  for u in "${cand[@]:-}"; do
    [[ -z "${u}" ]] && continue
    skip=0
    for v in "${cand[@]:-}"; do
      if [[ -z "${v}" || "${v}" == "${u}" ]]; then continue; fi
      if product_anc "${u}" "${v}"; then skip=1; break; fi
    done
    if [[ "${skip}" -eq 0 ]]; then out+=("${u}"); fi
  done
  OUT_PARENTS="$(IFS=,; printf '%s' "${out[*]:-}")"
}

# Orphan histories carry their seed in the root commit subject: `pin <sha>`.
orphan_base() { # <tag> -> 0 + OUT_BASE when the history is orphaned
  local t="$1" leaf sha root subj hex u
  OUT_BASE=""
  for leaf in InfiniCore InfiniLM; do
    sha="${PIN_IC[$t]:-}"
    if [[ "${leaf}" == InfiniLM ]]; then sha="${PIN_IL[$t]:-}"; fi
    [[ -z "${sha}" ]] && continue
    root="$(git -C "$(leaf_repo "${leaf}")" rev-list --max-parents=0 "${sha}" 2>/dev/null | head -n1 || true)"
    [[ -z "${root}" ]] && continue
    subj="$(git -C "$(leaf_repo "${leaf}")" log -1 --format=%s "${root}" 2>/dev/null || true)"
    # Roots in the wild use abbreviated SHAs, so accept 7..40 and resolve.
    [[ "${subj}" =~ ^pin[[:space:]]+([0-9a-f]{7,40})$ ]] || continue
    hex="${BASH_REMATCH[1]}"
    hex="$(git -C "$(leaf_repo "${leaf}")" rev-parse --verify "${hex}^{commit}" 2>/dev/null || true)"
    [[ -n "${hex}" ]] || continue
    for u in "${TAGS[@]}"; do
      if [[ "${leaf}" == InfiniCore && "${PIN_IC[$u]:-}" == "${hex}" ]]; then OUT_BASE="${u}"; return 0; fi
      if [[ "${leaf}" == InfiniLM && "${PIN_IL[$u]:-}" == "${hex}" ]]; then OUT_BASE="${u}"; return 0; fi
    done
  done
  return 1
}

depth_of() { # <tag> -> OUT_DEPTH
  local t="$1" u d best=0
  if [[ -n "${DEPTH[$t]:-}" ]]; then OUT_DEPTH="${DEPTH[$t]}"; return 0; fi
  for u in "${REPS[@]}"; do
    if [[ "${u}" == "${t}" ]]; then continue; fi
    if same_pin "${u}" "${t}"; then continue; fi
    if ! product_anc "${u}" "${t}"; then continue; fi
    depth_of "${u}"
    d="${OUT_DEPTH}"
    if (( d + 1 > best )); then best=$(( d + 1 )); fi
  done
  DEPTH["${t}"]="${best}"
  OUT_DEPTH="${best}"
}

stream_of() { # <tag> -> declared STREAM, empty when undeclared
  local f="${LINEAGE_DIR}/$1.env" s="" line
  if [[ -f "${f}" ]]; then
    while IFS= read -r line; do
      line="${line%$'\r'}"
      if [[ "${line}" == STREAM=* ]]; then s="${line#STREAM=}"; break; fi
    done < "${f}"
  fi
  echo "${s}"
}

kind_of() { # <tag> -> pin | snapshot | fork | republish
  local t="$1" u
  for u in "${TAGS[@]}"; do
    if [[ "${u}" == "${t}" ]]; then break; fi
    if same_pin "${u}" "${t}"; then echo republish; return 0; fi
  done
  max_parents "${t}"
  if [[ -n "${OUT_PARENTS}" ]]; then echo pin; return 0; fi
  if orphan_base "${t}"; then echo snapshot; return 0; fi
  echo fork
}

# ---------------------------------------------------------------- commands

cmd_graph() {
  local fmt="${1:-}" t stream depth id label u p
  if [[ "${fmt}" != "--mermaid" ]]; then
    echo "pin lineage: ${#REPS[@]} pins, ${#TAGS[@]} tags"
    for t in "${REPS[@]}"; do
      stream="$(stream_of "${t}")"
      depth_of "${t}"; depth="${OUT_DEPTH}"
      printf '%-10s %2d  %s\n' "${stream}" "${depth}" "${t}"
    done | sort -k1,1 -k2,2n
    return 0
  fi
  echo "flowchart TD"
  local -A sg_seen=() sg_nodes=()
  local -a sg_order=()
  for t in "${REPS[@]}"; do
    stream="$(stream_of "${t}")"
    id="N${t//[!0-9A-Za-z]/}"
    label="${t}"
    for u in "${TAGS[@]}"; do
      if [[ "${u}" != "${t}" ]] && same_pin "${u}" "${t}"; then label="${label}<br/>= ${u}"; fi
    done
    if [[ -z "${sg_seen[${stream}]:-}" ]]; then
      sg_seen["${stream}"]=1
      sg_order+=("${stream}")
    fi
    sg_nodes["${stream}"]+="    ${id}[\"${label}\"]"$'\n'
  done
  for stream in "${sg_order[@]:-}"; do
    echo "  subgraph SG_${stream//[!0-9A-Za-z]/}[\"${stream}\"]"
    echo "    direction TB"
    printf '%s' "${sg_nodes[${stream}]}"
    echo "  end"
  done
  for t in "${REPS[@]}"; do
    id="N${t//[!0-9A-Za-z]/}"
    max_parents "${t}"
    local -a ps=()
    IFS=',' read -r -a ps <<< "${OUT_PARENTS:-}"
    for p in "${ps[@]:-}"; do
      [[ -z "${p}" ]] && continue
      echo "  N${p//[!0-9A-Za-z]/} --> ${id}"
    done
    if [[ -z "${OUT_PARENTS}" ]] && orphan_base "${t}"; then
      echo "  N${OUT_BASE//[!0-9A-Za-z]/} -. \"declared BASE\" .-> ${id}"
    fi
  done
}

cmd_explain() {
  local t="$1" u
  if [[ -z "${PIN_IC[$t]:-}" && -z "${PIN_IL[$t]:-}" ]]; then die "unknown or unpinned tag: ${t}"; fi
  echo "tag:    ${t}"
  echo "pin:    IC=${PIN_IC[$t]} IL=${PIN_IL[$t]}"
  echo "stream: $(stream_of "${t}")"
  echo "kind:   $(kind_of "${t}")"
  echo
  for u in "${TAGS[@]}"; do
    if [[ "${u}" != "${t}" ]] && same_pin "${u}" "${t}"; then
      echo "republished as: ${u}"
    fi
  done
  for u in "${REPS[@]}"; do
    if [[ "${u}" == "${t}" ]]; then continue; fi
    if product_anc "${u}" "${t}"; then echo "lineage: ${u}"; fi
  done
  if orphan_base "${t}"; then echo "base:   ${OUT_BASE} (declared, orphan history)"; fi
}

cmd_emit() {
  local t="$1" write="${2:-}" parents base kind body stream ef note v0 nl
  is_tag "${t}" || die "no such tag: ${t}"
  [[ -n "${PIN_IC[$t]:-}" ]] || die "tag has no pin vector: ${t} (list it in lineage/UNPINNED)"
  max_parents "${t}"; parents="${OUT_PARENTS}"
  base=""
  if [[ -z "${parents}" ]] && orphan_base "${t}"; then base="${OUT_BASE}"; fi
  # Newest commit this pin shares with the leaf's main branch. Empty when the
  # histories are unrelated (orphan seeds).
  local mb_ic mb_il
  mb_ic="$(git -C "$(leaf_repo InfiniCore)" merge-base "${PIN_IC[$t]}" "${MAIN_REF}" 2>/dev/null || true)"
  mb_il="$(git -C "$(leaf_repo InfiniLM)" merge-base "${PIN_IL[$t]}" "${MAIN_REF}" 2>/dev/null || true)"
  kind="$(kind_of "${t}")"
  if [[ -z "$(stream_of "${t}")" ]]; then
    echo "warning: STREAM is not derivable, set it by hand in lineage/${t}.env" >&2
  fi
  stream="$(stream_of "${t}")"
  # Hand-declared fields win over derivation when the file already exists.
  ef="${LINEAGE_DIR}/${t}.env"
  if [[ -f "${ef}" ]]; then
    v0="$(env_field "${ef}" STREAM)";       [[ -n "${v0}" ]] && stream="${v0}"
    v0="$(env_field "${ef}" PARENT)";       [[ -n "${v0}" ]] && parents="${v0}"
    v0="$(env_field "${ef}" BASE)";         [[ -n "${v0}" ]] && base="${v0}"
    v0="$(env_field "${ef}" MAIN_BASE_IC)"; [[ -n "${v0}" ]] && mb_ic="${v0}"
    v0="$(env_field "${ef}" MAIN_BASE_IL)"; [[ -n "${v0}" ]] && mb_il="${v0}"
    note="$(env_field "${ef}" NOTE)"
  fi
  local -a nlines=()
  local nm u nb r
  for nm in "${NESTED_NAMES[@]:-}"; do
    [[ -z "${nm}" ]] && continue
    u="${nm^^}"
    nlines+=("PIN_IC_${u}=${PIN_NESTED[${t}|${nm}]:-}")
    nb="$(env_field "${ef}" "MAIN_BASE_IC_${u}")"
    if [[ -z "${nb}" ]] && r="$(nested_repo "${nm}")"; then
      nb="$(git -C "${r}" merge-base "${PIN_NESTED[${t}|${nm}]:-}" "$(nested_main_ref "${nm}")" 2>/dev/null || true)"
    fi
    nlines+=("MAIN_BASE_IC_${u}=${nb}")
  done
  body="TAG=${t}
PIN_IC=${PIN_IC[$t]}
PIN_IL=${PIN_IL[$t]}
STREAM=${stream}
KIND=${kind}
PARENT=${parents}
BASE=${base}
MAIN_BASE_IC=${mb_ic}
MAIN_BASE_IL=${mb_il}"
  for nl in "${nlines[@]:-}"; do
    [[ -z "${nl}" ]] && continue
    body="${body}
${nl}"
  done
  body="${body}
NOTE=${note}"
  if [[ "${write}" == "--write" ]]; then
    mkdir -p "${LINEAGE_DIR}"
    printf '%s\n' "${body}" > "${LINEAGE_DIR}/${t}.env"
    echo "wrote lineage/${t}.env"
  else
    printf '%s\n' "${body}"
  fi
}

cmd_verify() {
  local t u f p base fails=0 unpinned=0 covered nm np r nb want
  local -a declared=()
  for t in "${TAGS[@]}"; do
    if is_unpinned "${t}"; then unpinned=$((unpinned + 1)); continue; fi
    f="${LINEAGE_DIR}/${t}.env"
    if [[ ! -f "${f}" ]]; then
      echo "MISSING  lineage/${t}.env"; fails=$((fails + 1)); continue
    fi
    if [[ "$(env_field "${f}" PIN_IC)" != "${PIN_IC[$t]}" ]]; then
      echo "PIN_IC   mismatch: ${t}"; fails=$((fails + 1))
    fi
    if [[ "$(env_field "${f}" PIN_IL)" != "${PIN_IL[$t]}" ]]; then
      echo "PIN_IL   mismatch: ${t}"; fails=$((fails + 1))
    fi
    if [[ -z "$(env_field "${f}" STREAM)" ]]; then
      echo "STREAM   empty: ${t}"; fails=$((fails + 1))
    fi
    case "$(env_field "${f}" KIND)" in
      pin|snapshot|fork|republish) ;;
      *) echo "KIND    invalid: ${t}"; fails=$((fails + 1)) ;;
    esac
    declared=()
    IFS=',' read -r -a declared <<< "$(env_field "${f}" PARENT)"
    for p in "${declared[@]:-}"; do
      [[ -z "${p}" ]] && continue
      if ! product_anc "${p}" "${t}"; then
        echo "PARENT  ${t}: ${p} is not an ancestor"; fails=$((fails + 1))
      fi
    done
    # maximality: every strict ancestor must sit below a declared parent
    for u in "${REPS[@]}"; do
      if [[ "${u}" == "${t}" ]]; then continue; fi
      if same_pin "${u}" "${t}"; then continue; fi
      if ! product_anc "${u}" "${t}"; then continue; fi
      covered=0
      for p in "${declared[@]:-}"; do
        if [[ -n "${p}" ]] && product_anc "${u}" "${p}"; then covered=1; break; fi
      done
      if [[ "${covered}" -eq 0 ]]; then
        echo "PARENT  ${t}: ${u} is an undeclared ancestor"; fails=$((fails + 1))
      fi
    done
    base="$(env_field "${f}" BASE)"
    if [[ -n "${base}" ]]; then
      if [[ ${#declared[@]} -gt 0 ]]; then
        echo "BASE    ${t}: only a root publication may declare a base"; fails=$((fails + 1))
      fi
      if ! orphan_base "${t}"; then
        echo "BASE    ${t}: history is not orphaned"; fails=$((fails + 1))
      elif [[ "${OUT_BASE}" != "${base}" ]]; then
        echo "BASE    ${t}: ${base} is not provable from the orphan root"; fails=$((fails + 1))
      fi
    fi
  # shared-with-main must be reproducible from the pin and MAIN_REF
  local mb_ic mb_il
  mb_ic="$(git -C "$(leaf_repo InfiniCore)" merge-base "${PIN_IC[$t]}" "${MAIN_REF}" 2>/dev/null || true)"
  mb_il="$(git -C "$(leaf_repo InfiniLM)" merge-base "${PIN_IL[$t]}" "${MAIN_REF}" 2>/dev/null || true)"
  if [[ -n "$(env_field "${f}" MAIN_BASE_IC)" && "$(env_field "${f}" MAIN_BASE_IC)" != "${mb_ic}" ]]; then
    echo "MAIN_B  ${t}: InfiniCore shared-with-main mismatch"; fails=$((fails + 1))
  fi
  if [[ -n "$(env_field "${f}" MAIN_BASE_IL)" && "$(env_field "${f}" MAIN_BASE_IL)" != "${mb_il}" ]]; then
    echo "MAIN_B  ${t}: InfiniLM shared-with-main mismatch"; fails=$((fails + 1))
  fi
  # nested dependency leaves of InfiniCore: pins must match its tree
  for nm in "${NESTED_NAMES[@]:-}"; do
    [[ -z "${nm}" ]] && continue
    u="${nm^^}"
    np="$(git -C "$(leaf_repo InfiniCore)" rev-parse "${PIN_IC[${t}]}:${NESTED_PREFIX}/${nm}" 2>/dev/null || true)"
    if [[ "$(env_field "${f}" "PIN_IC_${u}")" != "${np}" ]]; then
      echo "PIN_IC_${u} mismatch: ${t}"; fails=$((fails + 1))
    fi
    r="$(nested_repo "${nm}")" || continue
    nb="$(git -C "${r}" merge-base "${np}" "$(nested_main_ref "${nm}")" 2>/dev/null || true)"
    want="$(env_field "${f}" "MAIN_BASE_IC_${u}")"
    if [[ -n "${want}" && "${want}" != "${nb}" ]]; then
      echo "MAIN_B  ${t}: ${nm} shared-with-main mismatch"; fails=$((fails + 1))
    fi
  done
  done
  for f in "${LINEAGE_DIR}"/*.env; do
    [[ -e "${f}" ]] || continue
    t="$(basename "${f}" .env)"
    if ! is_tag "${t}"; then echo "ORPHAN   lineage/${t}.env has no tag"; fails=$((fails + 1)); fi
  done
  for nm in "${NESTED_NAMES[@]:-}"; do
    [[ -z "${nm}" ]] && continue
    nested_repo "${nm}" >/dev/null || echo "NESTED  ${nm}: repo unavailable; shared-with-main checks skipped"
  done
  echo "tags: ${#TAGS[@]}  unpinned: ${unpinned}  failures: ${fails}"
  [[ "${fails}" -eq 0 ]]
}

cmd_index() {
  local t f first=1 nest nmu v first2
  {
    echo "{"
    echo '  "generated_by": "scripts/lineage.sh index",'
    echo '  "tags": {'
    for t in "${TAGS[@]}"; do
      f="${LINEAGE_DIR}/${t}.env"
      [[ -f "${f}" ]] || continue
      if [[ "${first}" -eq 0 ]]; then echo ','; fi
      first=0
      nest=""
      first2=1
      for nmu in "${NESTED_NAMES[@]:-}"; do
        [[ -z "${nmu}" ]] && continue
        v="$(env_field "${f}" "PIN_IC_${nmu^^}")"
        [[ -z "${v}" ]] && continue
        if [[ "${first2}" -eq 0 ]]; then nest+=', '; fi
        first2=0
        nest+="\"${nmu}\": \"${v}\""
      done
      printf '    "%s": {"stream": "%s", "kind": "%s", "parent": "%s", "base": "%s", "nested": {%s}}' \
        "${t}" "$(env_field "${f}" STREAM)" "$(env_field "${f}" KIND)" \
        "$(env_field "${f}" PARENT)" "$(env_field "${f}" BASE)" "${nest}"
    done
    echo
    echo "  }"
    echo "}"
  } > "${INDEX_FILE}"
  echo "wrote $(basename "${INDEX_FILE}")"
}

cmd_emit_all() {
  local t
  for t in "${TAGS[@]}"; do
    if [[ -z "${PIN_IC[$t]:-}" ]]; then
      echo "skipped (no pin vector): ${t}"
      continue
    fi
    cmd_emit "${t}" "--write"
  done
}

load

case "${1:-}" in
  graph) cmd_graph "${2:-}" ;;
  explain) shift; cmd_explain "${1:?usage: lineage.sh explain <tag>}" ;;
  emit) shift
    if [[ "${1:-}" == "--all" ]]; then cmd_emit_all
    else cmd_emit "${1:?usage: lineage.sh emit <tag> [--write]}" "${2:-}"; fi ;;
  verify) cmd_verify ;;
  index) cmd_index ;;
  *) sed -n '2,19p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
