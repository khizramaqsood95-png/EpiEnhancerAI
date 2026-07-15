#!/usr/bin/env python3
"""
EpiEnhancerAI.py

Single entry point for the enhancer-annotation pipeline. Pick which stage to
run with the first command-line argument:

    python3 EpiEnhancerAI.py preprocessing     --output_path ... --input_file ... [--normalisation min_max] [--chrom_sizes hg38.chrom.sizes.txt] [--binsize 100]
    python3 EpiEnhancerAI.py model_training    --train_file ... --test_file ... --model_name LR --output_dir ...
    python3 EpiEnhancerAI.py model_prediction  --test_file ... --model_name LR --model_file ... --output_dir ...
    python3 EpiEnhancerAI.py assembly          --enhancer_annotation_csv ... --output_path ... [--threshold 0.8] [--gap 500] [--bin_size 100]

The subcommand name is case-insensitive, so "preprocessing", "Preprocessing",
"Model_training", "MODEL_TRAINING", "Model_prediction", "Assembly", etc. all work.
"""

import argparse
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig
import pyranges as pr
import warnings

warnings.filterwarnings("ignore")


# =============================================================================
# =========================== PRE-PROCESSING STAGE ===========================
# =============================================================================

META_COLS = ["seqnames", "start", "end", "width", "strand"]

# Row types expected in the marks CSV (3rd column: "type")
TRACK_TYPES = ["input", "label"]


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------
def load_chrom_sizes_from_file(chrom_sizes_file):
    chrom_sizes = {}
    with open(chrom_sizes_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chrom, size = line.split()[:2]
            chrom_sizes[chrom] = int(size)
    return chrom_sizes


def empty_tiled_genome_from_chrom_sizes(chrom_sizes_file, binsize=100, chroms=None, single_point=False):

    chrom_sizes = load_chrom_sizes_from_file(chrom_sizes_file)
    if chroms is None:
        chroms = list(chrom_sizes.keys())

    rows = []
    total = len(chroms)

    for idx, chrom in enumerate(chroms, start=1):
        if chrom not in chrom_sizes:
            continue
        print(f"Building tile bins for chromosome {idx}/{total}: {chrom}")
        length = chrom_sizes[chrom]
        for start_1based in range(1, length + 1, binsize):
            start = start_1based - 1
            end = start + 1 if single_point else min(start + binsize, length)
            rows.append((chrom, start, end, "*"))

    # PyRanges requires these exact capitalized names to construct the object.
    pyranges_df = pd.DataFrame(rows, columns=["Chromosome", "Start", "End", "Strand"])
    genome_tiled = pr.PyRanges(pyranges_df)

    genome_df = genome_tiled.df.reset_index(drop=True).rename(columns={
        "Chromosome": "seqnames",
        "Start": "start",
        "End": "end",
        "Strand": "strand",
    })
    genome_df.insert(3, "width", genome_df["end"] - genome_df["start"])
    genome_df = genome_df[META_COLS]

    return genome_tiled, genome_df


# ---------------------------------------------------------------------------
# BED reading (used for label / STARR-style peak files)
# ---------------------------------------------------------------------------
def read_bed_as_pyranges(bed_path):
    df = pd.read_csv(bed_path, sep="\t", header=None, comment="#")
    df = df.iloc[:, :3].copy()
    # PyRanges requires these exact capitalized names.
    df.columns = ["Chromosome", "Start", "End"]
    df["Start"] = df["Start"].astype(int)
    df["End"] = df["End"].astype(int)
    return pr.PyRanges(df)


def _is_bed_file(path):
    suffixes = "".join(Path(path).suffixes).lower()
    return suffixes.endswith(".bed") or suffixes.endswith(".bed.gz")


def read_bigwig_signal(bw_path, tiled_df, binsize, agg="mean", threshold=None):

    bw = pyBigWig.open(str(bw_path))
    bw_chroms = bw.chroms() or {}

    out = np.zeros(len(tiled_df), dtype=float)

    for chrom, chrom_df in tiled_df.groupby("seqnames", sort=False):
        print("following chromosome is being processing: ", chrom)
        idx = chrom_df.index.to_numpy()
        starts = chrom_df["start"].to_numpy()
        ends = chrom_df["end"].to_numpy()

        if chrom not in bw_chroms:
            continue  

        bw_chrom_len = int(bw_chroms[chrom])

        keep = starts < bw_chrom_len
        if not keep.any():
            continue
        idx, starts, ends = idx[keep], starts[keep], ends[keep]
        ends = np.minimum(ends, bw_chrom_len)

        intervals = bw.intervals(chrom, 0, bw_chrom_len)
        if intervals:
            iv_starts = np.array([iv[0] for iv in intervals])
            iv_ends = np.array([iv[1] for iv in intervals])
            iv_vals = np.array([iv[2] for iv in intervals], dtype=float)

            pos = np.searchsorted(iv_starts, ends, side="left") - 1
            valid = pos >= 0
            overlaps = np.zeros(len(idx), dtype=bool)
            overlaps[valid] = iv_ends[pos[valid]] > starts[valid]

            chosen = np.where(overlaps)[0]
            out[idx[chosen]] = iv_vals[pos[chosen]]

        if len(idx) > 0:
            out[idx[-1]] = 0.0

    bw.close()

    if threshold is not None:
        out = (out > threshold).astype(float)

    return out


def starr_binary_bed(genome_tiled, tiled_df, bed_path):

    peaks = read_bed_as_pyranges(bed_path)

    overlap_pr = genome_tiled.count_overlaps(peaks, overlap_col="NumOverlaps")
    overlap_df = overlap_pr.df[["Chromosome", "Start", "End", "NumOverlaps"]].rename(columns={
        "Chromosome": "seqnames",
        "Start": "start",
        "End": "end",
    })

    merged = tiled_df[["seqnames", "start", "end"]].merge(
        overlap_df, on=["seqnames", "start", "end"], how="left"
    )
    merged["NumOverlaps"] = merged["NumOverlaps"].fillna(0)

    return (merged["NumOverlaps"].to_numpy() > 0).astype(int)


def starr_binary_bigwig(tiled_df, bw_path, binsize, threshold=0.0):
    return read_bigwig_signal(bw_path, tiled_df, binsize=binsize, threshold=threshold)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def zero_to_na(x):
    return x.mask(x == 0)


def normalize_min_max(df):
    out = df.copy()
    for c in out.columns:
        mn = out[c].min()
        mx = out[c].max()
        out[c] = (out[c] - mn) / (mx - mn) if mx != mn else 0.0
    return out


def normalize_quantile(df, q_low=0.01, q_high=0.99):
    out = df.copy()
    for c in out.columns:
        mn = out[c].quantile(q_low)
        mx = out[c].quantile(q_high)
        out[c] = (out[c] - mn) / (mx - mn) if mx != mn else 0.0
    return out


def _dedupe_colname(name, used):
    """Guard against two rows in the CSV accidentally sharing the same feature name."""
    if name not in used:
        used.add(name)
        return name
    n = 2
    while f"{name}_{n}" in used:
        n += 1
    new_name = f"{name}_{n}"
    used.add(new_name)
    print(f"Warning: duplicate column name '{name}' in CSV, renamed to '{new_name}'")
    return new_name


def _process_input_track_list(tiled_df, genome_df, entries, binsize, used_colnames, threshold=None):
    """Reads a list of {"path", "mark"} bigWig entries (type=input) and adds one column per entry."""
    added_cols = []
    for entry in entries:
        path = Path(entry["path"])
        colname = _dedupe_colname(entry["mark"], used_colnames)
        if not path.exists():
            print(f"Warning: file not found, skipping '{colname}': {path}")
            continue
        print(f"Reading input track ({colname}): {path}")
        genome_df[colname] = read_bigwig_signal(path, tiled_df, binsize=binsize, threshold=threshold)
        added_cols.append(colname)
    return added_cols


def _process_label_track_list(genome_tiled, tiled_df, genome_df, entries, binsize, used_colnames):
    """Reads a list of {"path", "mark"} entries (type=label) using the STARR_seq binary logic:
    .bed / .bed.gz files are handled as peak overlaps, anything else as a thresholded bigWig."""
    added_cols = []
    for entry in entries:
        path = Path(entry["path"])
        colname = _dedupe_colname(entry["mark"], used_colnames)
        if not path.exists():
            print(f"Warning: file not found, skipping '{colname}': {path}")
            continue
        if _is_bed_file(path):
            print(f"Reading label peak file ({colname}): {path}")
            genome_df[colname] = starr_binary_bed(genome_tiled, tiled_df, path)
        else:
            print(f"Reading label BigWig ({colname}): {path}")
            genome_df[colname] = starr_binary_bigwig(tiled_df, path, binsize=binsize, threshold=0.0)
        added_cols.append(colname)
    return added_cols


# ---------------------------------------------------------------------------
# Main pre-processing pipeline
# ---------------------------------------------------------------------------
def build_genome_tiled(
    output_path,
    normalisation,
    chrom_sizes_file,
    tracks,
    binsize=100,
):

    print("Pre-processing...")
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    genome_tiled, genome_df = empty_tiled_genome_from_chrom_sizes(
        chrom_sizes_file=chrom_sizes_file,
        binsize=binsize,
        single_point=False,
    )

    tiled_df = genome_df[["seqnames", "start", "end"]].copy()

    used_colnames = set()

    # ---- input tracks (continuous bigWig signal) ----
    input_col_names = []
    if tracks.get("input"):
        print("\nInput track processing")
        input_col_names = _process_input_track_list(
            tiled_df, genome_df, tracks["input"], binsize, used_colnames
        )
        genome_df.to_csv(output_path / "FeatureMatrix_input.csv", index=False, na_rep="NA")

    # ---- label tracks (STARR_seq binary logic) ----
    if tracks.get("label"):
        print("\nLabel track processing (STARR_seq)")
        _process_label_track_list(
            genome_tiled, tiled_df, genome_df, tracks["label"], binsize, used_colnames
        )
        genome_df.to_csv(output_path / "FeatureMatrix_label.csv", index=False, na_rep="NA")

    non_meta = [c for c in genome_df.columns if c not in META_COLS]

    filename = ""
    # ---- Normalisation + NA-masked outputs ----
    if normalisation == "min_max":

        genome_df[non_meta] = normalize_min_max(genome_df[non_meta])
        genome_df.to_csv(output_path / "FeatureMatrix_norm_tiled.csv", index=False, na_rep="NA")

        na_df = genome_df.copy()
        if input_col_names:
            na_df[input_col_names] = na_df[input_col_names].apply(zero_to_na)
        na_df.to_csv(output_path / "FeatureMatrix_norm_tiled_NA.csv", index=False, na_rep="NA")
        print("\nData has been saved in norm file!")
        filename = output_path / "FeatureMatrix_norm_tiled_NA.csv"

    elif normalisation == "quantile":
        genome_df[non_meta] = normalize_quantile(genome_df[non_meta])
        genome_df.to_csv(output_path / "FeatureMatrix_quantile_0.01_0.99_tiled.csv", index=False, na_rep="NA")

        na_df = genome_df.copy()
        if input_col_names:
            na_df[input_col_names] = na_df[input_col_names].apply(zero_to_na)
        na_df.to_csv(output_path / "FeatureMatrix_quantile_0.01_0.99_tiled_NA.csv", index=False, na_rep="NA")
        print("\nData has been saved in quantile file!")
        filename = output_path / "FeatureMatrix_quantile_0.01_0.99_tiled_NA.csv"

    else:
        genome_df.to_csv(output_path / "FeatureMatrix_tiled.csv", index=False, na_rep="NA")
        print("\nData has been saved in tiled file!")
        filename = output_path / "FeatureMatrix_tiled.csv"

    return genome_df, filename, output_path


def build_stratified_samples(
    tiled_csv_path,
    output_path,
    label_col="STARR",
    total_samples=500_000,
    seed=1509,
):
    print("Data splitting in Train and Test file...")
    tiled_csv_path = Path(tiled_csv_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    genome_tiled = pd.read_csv(tiled_csv_path)

    chrom_order = genome_tiled["seqnames"].unique()
    chrom_counts = genome_tiled["seqnames"].value_counts().reindex(chrom_order)
    total_bins = len(genome_tiled)
    chr_ratios = chrom_counts / total_bins

    positive_bins = genome_tiled[genome_tiled[label_col] == 1]
    negative_bins = genome_tiled[genome_tiled[label_col] == 0]

    pos_frac_overall = len(positive_bins) / total_bins
    neg_frac_overall = len(negative_bins) / total_bins

    pos_num = (chr_ratios * pos_frac_overall * total_samples).round().astype(int)
    neg_num = (chr_ratios * neg_frac_overall * total_samples).round().astype(int)

    rng = np.random.default_rng(seed)

    input_parts = []
    holdout_parts = []
    chosen_indices = []   # to track all chosen indices (input)
    holdout_indices = []  # to track all holdout indices

    def sample_chrom(bins_df, n_take, chrom):
        chrom_bins = bins_df[bins_df["seqnames"] == chrom]
        n_available = len(chrom_bins)
        if n_take <= 0 or n_available == 0:
            return None, None, None, None
        if n_take > n_available:
            raise ValueError(
                f"Requested {n_take} bins from {chrom} but only {n_available} "
                f"are available."
            )

        chosen_idx = rng.choice(chrom_bins.index.to_numpy(), size=n_take, replace=False)
        chosen = chrom_bins.loc[chosen_idx]

        remaining = chrom_bins.drop(index=chosen_idx)
        n_remaining_available = len(remaining)
        if n_take > n_remaining_available:
            raise ValueError(
                f"Not enough remaining {chrom} bins to build a same-sized "
                f"holdout set ({n_take} requested, {n_remaining_available} left)."
            )
        holdout_idx = rng.choice(remaining.index.to_numpy(), size=n_take, replace=False)
        holdout = remaining.loc[holdout_idx]

        return chosen, holdout, chosen_idx, holdout_idx

    # Negative bins first
    for chrom in negative_bins["seqnames"].unique():
        n_take = int(neg_num.get(chrom, 0))
        chosen, holdout, chosen_idx, holdout_idx = sample_chrom(negative_bins, n_take, chrom)
        if chosen is not None:
            input_parts.append(chosen)
            holdout_parts.append(holdout)
            chosen_indices.extend(chosen_idx)
            holdout_indices.extend(holdout_idx)

    # Positive bins second
    for chrom in positive_bins["seqnames"].unique():
        n_take = int(pos_num.get(chrom, 0))
        chosen, holdout, chosen_idx, holdout_idx = sample_chrom(positive_bins, n_take, chrom)
        if chosen is not None:
            input_parts.append(chosen)
            holdout_parts.append(holdout)
            chosen_indices.extend(chosen_idx)
            holdout_indices.extend(holdout_idx)

    H9_input_500k = pd.concat(input_parts, ignore_index=True)
    H9_holdout_500k = pd.concat(holdout_parts, ignore_index=True)

    # ---- unseen data ----
    used_indices = set(chosen_indices) | set(holdout_indices)
    leftover_indices = set(genome_tiled.index) - used_indices
    H9_unseen_500k = genome_tiled.loc[list(leftover_indices)]

    # Save all three files
    H9_input_500k.to_csv(output_path / "FeatureMatrix_input_500k.csv", index=False, na_rep="NA")     # Train file
    H9_holdout_500k.to_csv(output_path / "FeatureMatrix_holdout_500k.csv", index=False, na_rep="NA")  # Test file
    H9_unseen_500k.to_csv(output_path / "FeatureMatrix_unseen_500k.csv", index=False, na_rep="NA")    # unseen genome file

    print(f"Input set:        {len(H9_input_500k)} rows")
    print(f"Holdout set:      {len(H9_holdout_500k)} rows")
    print(f"Unseen (leftover) set: {len(H9_unseen_500k)} rows")

    return H9_input_500k, H9_holdout_500k, H9_unseen_500k


def read_preprocessing_params(csv_path):
    """Read the marks CSV, which only lists which marks to process.
    Expected columns (in order): Feature, path, type (input|label)."""

    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    feature_col, path_col, type_col = df.columns[0], df.columns[1], df.columns[2]

    tracks = {t: [] for t in TRACK_TYPES}

    for _, row in df.iterrows():
        feature = str(row[feature_col]).strip()

        path = row[path_col]
        path = None if pd.isna(path) else str(path).strip()

        track_type = row[type_col]
        track_type = None if pd.isna(track_type) else str(track_type).strip().lower()

        if not path:
            print(f"Warning: '{feature}' row in {csv_path} has no file path, skipping")
            continue

        if track_type not in TRACK_TYPES:
            print(f"Warning: unrecognised type '{track_type}' for '{feature}' in {csv_path}, skipping")
            continue

        tracks[track_type].append({"path": path, "mark": feature})

    return tracks


def run_preprocessing(args):
    tracks = read_preprocessing_params(args.params_csv)

    genome_tiled, tiled_csv_path, output_path = build_genome_tiled(
        output_path=args.output_path,
        normalisation=args.normalisation.lower(),
        chrom_sizes_file=args.chrom_sizes_file,
        tracks=tracks,
        binsize=args.binsize,
    )
    print("Splitting data into training, testing and unseen files...")
    build_stratified_samples(tiled_csv_path, output_path)
    print("Data splitting done...")


# =============================================================================
# =========================== MODEL TRAINING STAGE ===========================
# =============================================================================

def choose_model(model_name):
    """Map the --model_name value to the actual training/inference script filename."""
    mapping = {
        "LR": "LogisticRegression.py",  
        "XGB": "XGBoost.py",
        "CNN": "CNN.py",
        "FUZZY": "Fuzzy.py",
    }
    key = str(model_name).strip().upper()
    if key not in mapping:
        raise ValueError(
            f"Invalid --model_name '{model_name}'. Must be one of {sorted(mapping)}."
        )
    return mapping[key]


def run_model_training(args):
    print("Model Training Start...")

    do_train = True

    script_name = choose_model(args.model_name)
   
    no_training_flag = script_name.lower() in ("fuzzy.py", "cnn.py", "xgboost.py")

    if not args.train_file:
        raise ValueError("--train_file is required.")
    train_file = args.train_file
    test_file = args.test_file

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        base_dir = Path(__file__).resolve().parent
    except NameError:
        
        base_dir = Path.cwd()
    script_path = base_dir / script_name

    def _arg(value):
        return "" if value is None else str(value)

    if no_training_flag:
        run_args = [_arg(train_file), _arg(test_file), _arg(output_dir)]
    else:
        run_args = [_arg(do_train), _arg(train_file), _arg(test_file), _arg(output_dir), ""]

    print(f"Running: {script_name}")
    result = subprocess.run([sys.executable, str(script_path), *run_args])

    if result.returncode != 0:
        raise RuntimeError(f"{script_name} exited with a non-zero status ({result.returncode}).")

    print("Model Training Done...")


# =============================================================================
# ========================== MODEL PREDICTION STAGE ==========================
# =============================================================================


def choose_prediction_model(model_name):
    """Map the --model_name value to the actual *_pred.py inference script filename."""
    mapping = {
        "LG": "LogisticRegression_pred.py",
        "LR": "LogisticRegression_pred.py",   
        "XGB": "XGBoost_pred.py",
        "CNN": "CNN_pred.py",
        "FUZZY": "Fuzzy_pred.py",
    }
    key = str(model_name).strip().upper()
    if key not in mapping:
        raise ValueError(
            f"Invalid --model_name '{model_name}'. Must be one of {sorted(mapping)}."
        )
    return mapping[key]


def run_model_prediction(args):
    print("Model Prediction Start...")

    script_name = choose_prediction_model(args.model_name)
    is_fuzzy = script_name.lower() == "fuzzy_pred.py"

    test_file = args.test_file

    if is_fuzzy:
        model_file = None
        partition_file = args.partition_file
        rule_file = args.rule_file
        if not partition_file or not rule_file:
            raise ValueError(
                "--partition_file and --rule_file are required when --model_name is FUZZY."
            )
    else:
        model_file = args.model_file
        partition_file = None
        rule_file = None
        if not model_file:
            raise ValueError(
                "--model_file is required (unless --model_name is FUZZY)."
            )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        base_dir = Path(__file__).resolve().parent
    except NameError:
        
        base_dir = Path.cwd()
    script_path = base_dir / script_name

    def _arg(value):
        return "" if value is None else str(value)

    if not is_fuzzy:
        run_args = [_arg(test_file), _arg(output_dir), _arg(model_file)]
    else:
        run_args = [_arg(test_file), _arg(output_dir), _arg(partition_file), _arg(rule_file)]

    print(f"Running: {script_name}")
    result = subprocess.run([sys.executable, str(script_path), *run_args])

    if result.returncode != 0:
        raise RuntimeError(f"{script_name} exited with a non-zero status ({result.returncode}).")

    print("Model Prediction Done...")


# =============================================================================
# =============================== ASSEMBLY STAGE =============================
# =============================================================================

# The confidence/probability column is fixed.
CONF_COL = "Probabilities"


# ---------------------------------------------------------------------------
# Interval-merge
# ---------------------------------------------------------------------------
def reduce_intervals(starts, ends, min_gapwidth=1):

    pairs = sorted((int(s), int(e)) for s, e in zip(starts, ends))
    merged = []
    for s, e in pairs:
        if merged:
            gap = s - merged[-1][1] - 1
            if gap < min_gapwidth:
                if e > merged[-1][1]:
                    merged[-1] = (merged[-1][0], e)
                continue
        merged.append((s, e))
    return merged


# ---------------------------------------------------------------------------
# confidence score of every original bin overlapping it.
# ---------------------------------------------------------------------------
def conf_perc_avg(x_chrom_df, regions_df, conf_col, binsize=100):

    x_starts = x_chrom_df["start"].to_numpy()
    x_ends = x_chrom_df["end"].to_numpy()
    x_vals = x_chrom_df[conf_col].to_numpy(dtype=float)
    n = len(x_starts)

    sums = np.zeros(len(regions_df), dtype=float)
    j = 0
    for i, (rs, re) in enumerate(zip(regions_df["start"].to_numpy(), regions_df["end"].to_numpy())):
        total = 0.0
        while j < n and x_starts[j] < re:
            if x_ends[j] > rs:
                total += x_vals[j]
            if x_ends[j] <= re:
                j += 1
            else:
                break
        sums[i] = total

    out = regions_df.copy()
    out["sumconfperc1"] = sums
    widths = (out["end"] - out["start"]).to_numpy()
    out["avgconfperc1"] = sums / (widths / binsize)
    return out


def assign_domain_ids(regions_df, combined_intervals):

    domain_ids = np.full(len(regions_df), -1, dtype=int)
    k = 0
    n_combined = len(combined_intervals)
    for i, (rs, re) in enumerate(zip(regions_df["start"].to_numpy(), regions_df["end"].to_numpy())):
        while k < n_combined and combined_intervals[k][1] <= rs:
            k += 1
        if k < n_combined and combined_intervals[k][0] <= rs and combined_intervals[k][1] >= re:
            domain_ids[i] = k
        else:
            domain_ids[i] = k
    return domain_ids


# ---------------------------------------------------------------------------
# Equivalent of .p_glist_creation()
# ---------------------------------------------------------------------------
def p_glist_creation(chrom, df, threshold, gap, conf_col=CONF_COL, binsize=100):

    x_chrom = df[df["seqnames"] == chrom].sort_values("start").reset_index(drop=True)
    if x_chrom.empty:
        return []

    over_thresh = x_chrom[x_chrom[conf_col] >= threshold].sort_values("start").reset_index(drop=True)
    if over_thresh.empty:
        return []

    core_intervals = reduce_intervals(over_thresh["start"], over_thresh["end"], min_gapwidth=1)
    initial_regions = pd.DataFrame(core_intervals, columns=["start", "end"])
    initial_regions.insert(0, "seqnames", chrom)
    initial_regions = conf_perc_avg(x_chrom, initial_regions, conf_col=conf_col, binsize=binsize)

    # Broader domains: merge over-threshold bins allowing gaps up to `gap`.
    combined_intervals = reduce_intervals(over_thresh["start"], over_thresh["end"], min_gapwidth=gap)

    domain_ids = assign_domain_ids(initial_regions, combined_intervals)

    groups = []
    for domain_id in np.unique(domain_ids):
        group_df = initial_regions[domain_ids == domain_id].reset_index(drop=True)
        groups.append(group_df)
    return groups


# ---------------------------------------------------------------------------
# Equivalent of H9_glist.R's main loop
# ---------------------------------------------------------------------------
def build_glist(
    tiled_csv_path,
    output_path,
    threshold=0.8,
    gap=500,
    conf_col=CONF_COL,
    binsize=100,
    chrom_order=None,
):
    tiled_csv_path = Path(tiled_csv_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(tiled_csv_path)
    df.columns = [c.lower() if c.lower() in ("start", "end") else c for c in df.columns]
    if "seqnames" not in df.columns and "Chromosome" in df.columns:
        df = df.rename(columns={"Chromosome": "seqnames"})

    if chrom_order is None:
        chrom_order = list(df["seqnames"].unique())

    per_chrom_blocks = []
    for chrom in chrom_order:
        print(chrom)
        glist_temp = p_glist_creation(chrom, df, threshold=threshold, gap=gap, conf_col=conf_col, binsize=binsize)
        per_chrom_blocks.append(glist_temp)

    print("writing files")
    glist = []
    for block in reversed(per_chrom_blocks):
        glist.extend(block)

   
    with open(output_path / f"glist_{threshold}_{gap}bp.pkl", "wb") as f:
        pickle.dump(glist, f)

    
    flat_frames = []
    for domain_id, group_df in enumerate(glist):
        g = group_df.copy()
        g.insert(0, "domain_id", domain_id)
        flat_frames.append(g)
    flat_df = pd.concat(flat_frames, ignore_index=True) if flat_frames else pd.DataFrame()
    flat_df.to_csv(output_path / f"glist_{threshold}_{gap}bp.csv", index=False, na_rep="NA")

    print(f"{len(glist)} domains written "
          f"({sum(len(g) for g in glist)} core regions total).")

    return glist


# ---------------------------------------------------------------------------
def conf_perc_avg_regions(source_df, target_intervals, binsize):

    seqname = source_df["seqnames"].iloc[0] if len(source_df) else None
    src_starts = source_df["start"].to_numpy()
    src_ends = source_df["end"].to_numpy()
    src_sums = source_df["sumconfperc1"].to_numpy(dtype=float)

    rows = []
    for s, e in target_intervals:
        overlap_mask = (src_starts <= e) & (src_ends >= s)
        total = float(src_sums[overlap_mask].sum())
        width = e - s + 1
        avg = total / (width / binsize)
        rows.append((seqname, s, e, width, total, avg))

    return pd.DataFrame(rows, columns=["seqnames", "start", "end", "width", "sumconfperc1", "avgconfperc1"])


# ---------------------------------------------------------------------------
# Equivalent of Region_Growth()
# ---------------------------------------------------------------------------
def region_growth(domain_df, threshold, max_gap=500, binsize=100):

    x = domain_df.sort_values("start").reset_index(drop=True)
    n = len(x)
    n1 = n

    # Step 1: try merging the WHOLE domain into a single region.
    merged_all = reduce_intervals(x["start"], x["end"], min_gapwidth=max_gap + 1)
    all_reduced = conf_perc_avg_regions(x, merged_all, binsize)

    if len(all_reduced) != 1:
        print(f"Warning: expected the whole domain to merge into 1 region "
              f"with min_gapwidth={max_gap + 1}, got {len(all_reduced)}. "
              f"Using the first row only (matches R's implicit behaviour).")

    if all_reduced.loc[0, "avgconfperc1"] >= threshold:
        return all_reduced.iloc[[0]].reset_index(drop=True)

    baseline_intervals = list(zip(x["start"], x["end"]))
    all_results = [conf_perc_avg_regions(x, baseline_intervals, binsize)]
    cur_n = n
    while cur_n > 2:
        cur_n -= 1
        window_size = cur_n
        candidates = []
        for start_idx in range(0, n1 - window_size + 1):
            candidate = x.iloc[start_idx: start_idx + window_size]
            merged_candidate = reduce_intervals(candidate["start"], candidate["end"], min_gapwidth=max_gap + 1)
            reduced = conf_perc_avg_regions(candidate, merged_candidate, binsize)
            candidates.append(reduced)
        if candidates:
            all_results.append(pd.concat(candidates, ignore_index=True))

    all_results_df = pd.concat(all_results, ignore_index=True)
    above_threshold = all_results_df[all_results_df["avgconfperc1"] >= threshold].reset_index(drop=True)

    if above_threshold.empty:
        return above_threshold

    final_merged = reduce_intervals(above_threshold["start"], above_threshold["end"], min_gapwidth=1)
    reduced_regions = conf_perc_avg_regions(above_threshold, final_merged, binsize)
    return reduced_regions


# ---------------------------------------------------------------------------
# Read glist_creation output (either the .pkl or the flat .csv)
# ---------------------------------------------------------------------------
def read_glist(path):
    """Returns a list of DataFrames, one per domain, each with columns
    seqnames, start, end, sumconfperc1, avgconfperc1."""
    path = Path(path)
    if path.suffix == ".pkl":
        with open(path, "rb") as f:
            glist = pickle.load(f)
        return glist

    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "domain_id" not in df.columns:
        raise ValueError("Expected a 'domain_id' column in the CSV -- is this really "
                          "glist_creation's output?")
    glist = [g.drop(columns=["domain_id"]).reset_index(drop=True)
             for _, g in df.groupby("domain_id", sort=True)]
    return glist


# ---------------------------------------------------------------------------
# Equivalent of Enhancers_Assembly.R
# ---------------------------------------------------------------------------
def assemble_enhancers(
    glist_path,
    output_path,
    threshold=0.8,
    max_gap=500,
    binsize=100,
    output_name="Annotated_Merged_Enhancers",
):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    glist = read_glist(glist_path)

    no_check = [g for g in glist if len(g) == 1]
    to_check = [g for g in glist if len(g) > 1]

    no_check = [
        g.assign(width=g["end"] - g["start"] + 1)[
            ["seqnames", "start", "end", "width", "sumconfperc1", "avgconfperc1"]
        ]
        for g in no_check
    ]

    print("Number of regions not required to merged: ", len(no_check))
    print("Number of regions required to merged: ", len(to_check))
    grown_frames = []
    for domain_df in to_check:
        grown = region_growth(domain_df, threshold=threshold, max_gap=max_gap, binsize=binsize)
        if not grown.empty:
            grown_frames.append(grown)

    no_check_df = pd.concat(no_check, ignore_index=True) if no_check else pd.DataFrame(
        columns=["seqnames", "start", "end", "width", "sumconfperc1", "avgconfperc1"]
    )

    grown_df = pd.concat(grown_frames, ignore_index=True) if grown_frames else pd.DataFrame(
        columns=["seqnames", "start", "end", "width", "sumconfperc1", "avgconfperc1"]
    )

    final_df = pd.concat([no_check_df, grown_df], ignore_index=True)

    print("writing files")
    final_df.to_csv(output_path / f"{output_name}_{max_gap}bp.tsv", sep="\t", index=False, na_rep="NA")

    return final_df


def run_assembly(args):
    print("Enhancer Assembly start ...")
    print(f"Using tiled annotation CSV: {args.tiled_csv_path}")

    output_path = Path(args.output_path)

    print("Creating list of enhancer regions for Assembly...")
    build_glist(
        args.tiled_csv_path,
        output_path,
        threshold=args.threshold,
        gap=args.gap,
        conf_col=CONF_COL,
        binsize=args.binsize,
    )

    glist_path = output_path / f"glist_{args.threshold}_{args.gap}bp.pkl"

    print("Assembling the enhancer regions ...")
    assemble_enhancers(
        glist_path,
        output_path,
        threshold=args.threshold,
        max_gap=args.gap,
        binsize=args.binsize,
    )
    print("Enhancer Assembly Done!")


# =============================================================================
# ==================================== CLI ====================================
# =============================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        prog="EpiEnhancerAI.py",
        description="EpiEnhancerAI pipeline. Choose a step to run: "
                     "preprocessing, model_training, model_prediction, or assembly.",
    )
    subparsers = parser.add_subparsers(dest="step", required=True)

    # ---- preprocessing ----
    pp = subparsers.add_parser(
        "preprocessing",
        help="Tile the genome, build input/label tracks, normalise, and split into train/test/unseen sets.",
    )
    pp.add_argument("--output_path", required=True, help="Directory where output CSVs will be written.")
    pp.add_argument(
        "--normalisation", default="min_max", choices=["min_max", "quantile", "none"],
        help="Normalisation method to apply (default: min_max).",
    )
    pp.add_argument(
        "--chrom_sizes", dest="chrom_sizes_file", default="hg38.chrom.sizes.txt",
        help="Path to the chromosome sizes file (default: hg38.chrom.sizes.txt).",
    )
    pp.add_argument("--binsize", type=int, default=100, help="Tile bin size in bp (default: 100).")
    pp.add_argument(
        "--input_file", required=True, dest="params_csv",
        help="CSV file listing marks to process, with columns: Feature, path, type (input|label).",
    )

    # ---- model_training ----
    mt = subparsers.add_parser(
        "model_training",
        help="Train a chosen model (LG/LR, XGB, CNN, or FUZZY) on your data.",
    )
    mt.add_argument(
        "--train_file", required=True,
        help="Path to the training data file.",
    )
    mt.add_argument("--test_file", required=True, help="Path to the test/validation data file.")
    mt.add_argument(
        "--model_name", required=True,
        help="Model to train: LG/LR (Logistic Regression), XGB (XGBoost), CNN, or FUZZY.",
    )
    mt.add_argument("--output_dir", required=True, help="Directory for outputs.")

    # ---- model_prediction ----
    mp = subparsers.add_parser(
        "model_prediction",
        help="Run inference only, with a chosen model's *_pred.py script (LR_pred, XGBoost_pred, CNN_pred, fuzzy_pred).",
    )
    mp.add_argument("--test_file", required=True, help="Path to the test/inference data file.")
    mp.add_argument(
        "--model_name", required=True,
        help="Model to run: LG/LR (Logistic Regression), XGB (XGBoost), CNN, or FUZZY.",
    )
    mp.add_argument(
        "--model_file", default=None,
        help="Path to a saved model file. Required when --model_name is not FUZZY.",
    )
    mp.add_argument(
        "--partition_file", default=None,
        help="Path to the partition file. Required when --model_name is FUZZY.",
    )
    mp.add_argument(
        "--rule_file", default=None,
        help="Path to the rule file. Required when --model_name is FUZZY.",
    )
    mp.add_argument("--output_dir", required=True, help="Directory for outputs.")

    # ---- merging ----
    asm = subparsers.add_parser(
        "assembly",
        help="Build enhancer domain lists and assemble merged enhancer regions from a tiled annotation CSV.",
    )
    asm.add_argument(
        "--enhancer_annotation_csv", required=True, dest="tiled_csv_path",
        help="Path to the tiled annotation CSV (output of the preprocessing step).",
    )
    asm.add_argument("--output_path", required=True, help="Directory where output files will be written.")
    asm.add_argument(
        "--threshold", type=float, default=0.8, help="Confidence/probability threshold (default: 0.8).",
    )
    asm.add_argument(
        "--gap", type=int, default=500,
        help="Max gap in bp allowed when merging regions into domains (default: 500).",
    )
    asm.add_argument(
        "--bin_size", type=int, dest="binsize", default=100, help="Bin size in bp (default: 100).",
    )

    return parser


def main():
    # Allow the step name to be given in any case ("Model_training", "enhancer merging", etc.)
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        sys.argv[1] = sys.argv[1].lower()

    parser = build_parser()
    args = parser.parse_args()

    if args.step == "preprocessing":
        run_preprocessing(args)
    elif args.step == "model_training":
        run_model_training(args)
    elif args.step == "model_prediction":
        run_model_prediction(args)
    elif args.step == "assembly":
        run_assembly(args)


if __name__ == "__main__":
    main()
