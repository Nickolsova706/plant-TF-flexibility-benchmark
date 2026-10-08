# Preprocessing: Peak Cleaning, Motif Scanning, and Sequence Labeling

This directory contains the scripts used for preprocessing Arabidopsis thaliana transcription factor (TF) peak datasets prior to downstream DNA sequence, shape, flexibility, and machine-learning analyses.

The workflow starts from TF peak datasets in `.narrowPeak` format and produces positive/bound and negative/unbound DNA sequence datasets.

## Workflow

```text
Raw TF narrowPeak files
        ↓
00_generate_peak_sequences.py
        ↓
Peak FASTA sequences
        ↓
01_build_artifact_blacklist.sh
        ↓
global_artifacts.bed
        ↓
02_remove_artifacts.sh
        ↓
Cleaned TF narrowPeak files
        ↓
03_generate_tf_list.sh
        ↓
TF list
        ↓
04_collect_motifs.sh
        ↓
Selected MEME motifs
        ↓
05_rescue_missing_motifs.sh
        ↓
DREME-rescued motifs
        ↓
06_run_fimo.sh
        ↓
FIMO motif sites
        ↓
07_label_bound_unbound.sh
        ↓
Positive/bound and negative/unbound BED files
        ↓
08_extract_sequences.sh
        ↓
Positive and negative FASTA sequences
```

The resulting FASTA datasets are subsequently processed by the sequence-filtering workflow in:

```text
scripts/02_sequence_filtering/
```

---

## 0. Extract Peak Sequences

`00_generate_peak_sequences.py` extracts DNA sequences corresponding to genomic regions in a `.narrowPeak` file using the TAIR10 reference genome.

The script:

1. Reads the reference genome supplied with `-g`.
2. Stores chromosome sequences in memory.
3. Reads genomic peak coordinates from a `.narrowPeak` file supplied with `-p`.
4. Extracts the corresponding DNA sequence for each peak.
5. Writes the sequences to a FASTA file based on the input peak filename.

### Usage

```bash
python 00_generate_peak_sequences.py \
    -g /path/to/TAIR10_nuclear.fas \
    -p /path/to/chr1-5_GEM_events.narrowPeak
```

The output FASTA file is created in the current working directory using the input peak filename.

In the original workflow, this script was run across the TF peak datasets and the generated FASTA files were organized by TF.

---

## 1. Build the Global Artifact Blacklist

`01_build_artifact_blacklist.sh` identifies genomic regions that occur across a large number of TF peak datasets.

The script:

1. Collects genomic coordinates from all `.narrowPeak` files.
2. Records the source peak file for each region.
3. Sorts the genomic coordinates.
4. Uses `bedtools cluster` to group overlapping regions.
5. Counts the unique source peak datasets represented in each cluster.
6. Selects clusters represented in more than the specified threshold number of datasets.
7. Writes the corresponding genomic coordinates to `global_artifacts.bed`.

### Usage

```bash
./01_build_artifact_blacklist.sh \
    /path/to/00_peak_files \
    /path/to/output \
    144
```

The threshold used in this preprocessing workflow was **144 TF datasets**.

### Main output

```text
global_artifacts.bed
```

The artifact blacklist is used in the next step to remove regions shared across a large number of TF datasets.

---

## 2. Remove Artifact Regions

`02_remove_artifacts.sh` removes regions listed in `global_artifacts.bed` from each TF `.narrowPeak` file using:

```bash
bedtools intersect -v
```

The original peak files are retained, and cleaned files are written with the suffix:

```text
_clean.narrowPeak
```

### Usage

```bash
./02_remove_artifacts.sh \
    /path/to/00_peak_files \
    /path/to/global_artifacts.bed
```

### SRS7 and TRP2 Coordinate Correction

During the original preprocessing workflow, SRS7 and TRP2 contained negative genomic start coordinates.

For these two TF datasets only, records with negative start coordinates were removed before artifact subtraction. This correction was retained in `02_remove_artifacts.sh` to match the original preprocessing workflow.

---

## 3. Generate the TF List

`03_generate_tf_list.sh` creates a list of individual TF datasets from the TF family directory structure.

The script searches inside the TF family directories and writes the individual TF names to:

```text
my_219_tfs.txt
```

### Usage

```bash
./03_generate_tf_list.sh /path/to/00_peak_files
```

The generated list contains the TF datasets identified from the peak directory.

The generated TF list is a workflow output and is not stored in the repository as a tracked file.

---

## 4. Collect MEME Motifs

`04_collect_motifs.sh` collects MEME motif results corresponding to the TFs in the TF list.

For each TF, the script searches the supplied motif directory for the corresponding TF directory and copies its `meme_out` contents into the selected motif directory.

### Usage

```bash
./04_collect_motifs.sh \
    /path/to/motifs \
    /path/to/TF_list.txt \
    /path/to/02_selected_motifs
```

TFs for which a MEME motif directory cannot be found are reported as missing.

---

## 5. Rescue Missing Motifs Using DREME

Some TF datasets did not have a corresponding MEME motif result.

`05_rescue_missing_motifs.sh` retrieves the corresponding DREME motif results for these TFs.

The original workflow mapped the missing TF datasets to their corresponding TF families:

```text
At1g68670_colamp_a → G2like_tnt
HSF3_colamp_a      → HSF_tnt
LBD19_colamp_a     → LOBAS2_tnt
RAP26_colamp_a     → AP2EREBP_tnt
```

The DREME results were collected into:

```text
03_four_rescued_dreme/
```

and subsequently used as motif input.

### Usage

```bash
./05_rescue_missing_motifs.sh \
    /path/to/motifs \
    /path/to/03_four_rescued_dreme
```

---

## 6. Scan the Genome with FIMO

`06_run_fimo.sh` scans the TAIR10 genome for motif occurrences using the selected TF motifs.

The script uses:

```text
FIMO threshold: 5e-4
Maximum strand: enabled
Maximum stored scores: 1,000,000
```

FIMO searches for either:

```text
meme_m1.txt
```

or:

```text
dreme_d1.txt
```

for each TF.

### Usage

```bash
./06_run_fimo.sh \
    /path/to/02_selected_motifs \
    /path/to/TAIR10_nuclear.fas \
    /path/to/04_fimo_results
```

The FIMO results are stored separately for each TF.

---

## 7. Label Motif Sites as Bound or Unbound

`07_label_bound_unbound.sh` classifies motif occurrences based on their overlap with the corresponding cleaned TF peak regions.

For each FIMO motif occurrence:

1. A 100-bp window centered on the motif is generated.
2. The chromosome name is converted to lowercase.
3. The motif window is compared with the cleaned TF peak regions.
4. A motif window overlapping a TF peak is classified as **positive/bound**.
5. A motif window not overlapping a TF peak is classified as **negative/unbound**.

### Usage

```bash
./07_label_bound_unbound.sh \
    /path/to/04_fimo_results \
    /path/to/00_peak_files \
    /path/to/05_labeled_datasets
```

The resulting BED files are named:

```text
<TF>_positives.bed
<TF>_negatives.bed
```

---

## 8. Extract Positive and Negative DNA Sequences

`08_extract_sequences.sh` extracts DNA sequences from the positive and negative BED files using the TAIR10 genome.

The script:

1. Creates a lowercase chromosome-name version of the TAIR10 genome when needed.
2. Indexes the genome using `samtools faidx`.
3. Extracts positive sequences using `bedtools getfasta`.
4. Extracts negative sequences using `bedtools getfasta`.
5. Reports the number of BED intervals and successfully extracted FASTA sequences.

### Usage

```bash
./08_extract_sequences.sh \
    /path/to/TAIR10_nuclear.fas \
    /path/to/05_labeled_datasets \
    /path/to/06_extracted_fasta
```

For each TF, the output contains:

```text
<TF>_positives.fasta
<TF>_negatives.fasta
```

The script also reports sequences that could not be extracted, for example when genomic intervals fall outside the available chromosome coordinates.

---

## Sequence Filtering

After sequence extraction, three TF datasets were manually removed because they contained very few positive peaks relative to their negative datasets:

```text
NAP_colamp_a
BIM2_colamp
STOP1_colamp_a
```

The remaining extracted sequences were then filtered to retain only sequences containing the canonical DNA bases A, T, G, and C.

This filtering is performed by:

```text
scripts/02_sequence_filtering/keep_just_atgc.py
```

The script preserves the original extracted FASTA files and writes the ATGC-filtered sequences into a separate output directory.

---

## Data and Reproducibility

The scripts in this directory document the preprocessing workflow used for the analysis.

Large genomic datasets, TF peak files, genome files, motif databases, FIMO outputs, and generated FASTA datasets are not stored in this GitHub repository.

The repository contains the analysis scripts and documentation required to reproduce the workflow when the required input datasets and reference files are supplied.
