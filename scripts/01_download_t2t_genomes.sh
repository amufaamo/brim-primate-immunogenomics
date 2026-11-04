#!/usr/bin/env bash
set -euo pipefail

ROOT="${BRIM_DATA_DIR:-./data}"
assert_prefix="${BRIM_DATA_DIR:-./data}"
if [[ "$ROOT" != "$assert_prefix" ]]; then
  echo "FATAL: ROOT mismatch" >&2
  exit 1
fi

DATASETS="datasets"
mkdir -p "$ROOT/01_genomes" "$ROOT/logs"
cd "$ROOT/01_genomes"

cat > "$ROOT/01_genomes/t2t_manifest.tsv" << 'EOF'
assembly_id	species	accession
Human_GRCh38	Homo_sapiens	GCF_000001405.40
Human_CHM13	Homo_sapiens	GCF_009914755.1
Pan_troglodytes	Pan_troglodytes	GCF_028858775.2
Macaca_mulatta	Macaca_mulatta	GCF_049350105.2
Pan_paniscus	Pan_paniscus	GCF_029289425.2
Gorilla_gorilla	Gorilla_gorilla	GCF_029281585.2
Pongo_abelii	Pongo_abelii	GCF_028885655.2
Pongo_pygmaeus	Pongo_pygmaeus	GCF_028885625.2
Symphalangus_syndactylus	Symphalangus_syndactylus	GCF_028878055.3
EOF

download_one() {
  local aid="$1"
  local sp="$2"
  local acc="$3"
  local outdir="$ROOT/01_genomes/$aid"
  local done_flag="$outdir/.done"
  if [[ -f "$done_flag" ]]; then
    echo "[SKIP] $aid ($acc) already verified"
    return 0
  fi
  mkdir -p "$outdir"
  echo "[$(date +%H:%M:%S)] START $aid ($acc)"
  "$DATASETS" download genome accession "$acc" --include genome,gff3,seq-report --filename "$outdir/${aid}.zip" --no-progressbar
  unzip -q -o "$outdir/${aid}.zip" -d "$outdir"
  (
    cd "$outdir"
    md5sum -c md5sum.txt > md5_check.log
  )
  local fna
  fna=$(ls "$outdir"/ncbi_dataset/data/"$acc"/*_genomic.fna | head -n 1)
  if [[ ! -s "$fna" ]]; then
    echo "FATAL: genomic fna missing for $aid" >&2
    return 1
  fi
  touch "$done_flag"
  echo "[$(date +%H:%M:%S)] DONE  $aid ($acc) -> $fna"
}

export -f download_one
export ROOT DATASETS

tail -n +2 "$ROOT/01_genomes/t2t_manifest.tsv" | xargs -P 3 -n 3 bash -c 'download_one "$1" "$2" "$3"' _

echo "[$(date +%H:%M:%S)] ALL_9_GENOMES_DONE"
