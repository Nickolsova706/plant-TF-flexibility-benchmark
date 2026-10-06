# Preprocessing: Peak Cleaning and Artifact Removal

This directory contains the scripts used for the initial preprocessing of transcription factor (TF) peak datasets.

## Workflow

The preprocessing consists of two main steps:

```text
Raw TF narrowPeak files
        ↓
01_build_artifact_blacklist.sh
        ↓
global_artifacts.bed
        ↓
02_remove_artifacts.sh
        ↓
Cleaned TF narrowPeak files
```

## 1. Build the global artifact blacklist

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
./01_build_artifact_blacklist.sh <peak_dir> <output_dir> <threshold>
```

Example:

```bash
./01_build_artifact_blacklist.sh \
    /path/to/00_peak_files \
    /path/to/output \
    144
```

The threshold used in this preprocessing workflow was **144 datasets**.

### Main output

```text
global_artifacts.bed
```

## 2. Remove artifact regions

`02_remove_artifacts.sh` removes regions listed in `global_artifacts.bed` from every TF `.narrowPeak` file.

This is performed using:

```bash
bedtools intersect -v
```

The original peak files are retained, and cleaned files are written with the suffix:

```text
_clean.narrowPeak
```

### Usage

```bash
./02_remove_artifacts.sh <peak_base_dir> <artifact_bed>
```

Example:

```bash
./02_remove_artifacts.sh \
    /path/to/00_peak_files \
    /path/to/global_artifacts.bed
```

## Important

The scripts operate on the peak files and artifact blacklist but do not require the large datasets to be stored inside this GitHub repository.

Large genomic datasets and generated intermediate files should remain outside the repository.

The scripts are intended to document and reproduce the preprocessing workflow used for the analysis.
