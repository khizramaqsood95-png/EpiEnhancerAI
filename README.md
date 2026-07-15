# EpiEnhancerAI
Explainable AI identifies H3K18ac as a new marker of active enhancers


Contact: K.maqsood@qmul.ac.uk
You can find the updated version of EpiEnhancerAI on GitHub: https://github.com/khizramaqsood95-png/EpiEnhancerAI


A Python package for annotating enhancers from epigenetic data using AI/ML models
(Logistic Regression, XGBoost, CNN, and Type-2 Fuzzy Logic System). 
The package runs as four steps 
1- Pre-processing, 
2- Model Training (optional),
3- Model Prediction,
4- Enhancer merging
all driven from a single command-line script, `EpiEnhancerAi.py`.

---

## Requirements

Python 3.11 and above. Anaconda3 is recommended, especially for PyTorch (CNN model).

### PyTorch environment


# CPU only
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# OR with CUDA 12.1 GPU support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia


### Python libraries


pip install pybigwig
pip install pyranges
pip install pyfaidx
python -m pip install git+https://github.com/Fuminides/ex-fuzzy
python -m pip install imbalanced-learn
pip install xgboost
pip install scikit-learn
pip install pandas
pip install numpy


If you hit an `xgboost`/`scikit-learn` compatibility error (e.g. `AttributeError:'super' object has no attribute '__sklearn_tags__'`), upgrade xgboost:
`pip install -U xgboost` (requires xgboost >= 2.1.3).

### Required files alongside `EpiEnhancerAI.py`

The Model Training step calls out to the model-specific scripts below, so make
sure they sit in the same folder as `Enhancer_annotation.py`:

- `LogisticRegression.py`
- `XGBoost.py`
- `CNN.py`
- `Fuzzy.py`

---

## Pipeline Overview

The package consists of three steps, run in order:

1. Pre-processing: reads bigWig files (epigenetic tracks) and a BED
   file (STARR-seq annotation) and tiles the genome into 100bp bins to build
   the feature matrix required to train the AI/ML models. Also splits the
   matrix into train / holdout (test) / unseen sets.
2. Model Training takes the matrix from Pre-processing and trains
   (or runs inference with) an AI/ML model: LR, XGBoost, CNN, or Type-2 FLS
   (Fuzzy).
3. Model Prediction: takes the matrix from Pre-processing and predicts
   The enhancers using AI/ML models: LR, XGBoost, CNN, or Type-2 FLS
   (Fuzzy).
4. Enhancer merging takes the annotated-enhancer output of Model
   Training and merges adjacent/nearby 100bp bins back into full-length
   enhancer regions.

All four steps are run through the same script, choosing the step as the
first argument (case-insensitive: `preprocessing`, `model_training`, `model_prediction`
`enhancer_merging` all work):


python3 Enhancer_annotation.py <step> [options]


---

## 1. Pre-processing

Reads a CSV listing which epigenetic feature/track files to process, tiles
the genome using a chromosome-sizes file, builds the feature matrix, and
writes train/holdout/unseen splits.

### Features CSV (`--params_csv`)

This is the only file-based input for this step. It lists the epigenetic
tracks to include, with three columns:

| Column    | Description                                                                 |
|-----------|------------------------------------------------------------------------------|
| `Feature` | Name of the feature (becomes the output column name), e.g. `H3K27ac`, `ATAC`, `CpG`, `STARR` |
| `path`    | Full path to the track file (`.bigWig` or `.bed`/`.bed.gz`)                  |
| `type`    | Either `input` (continuous signal track, e.g. ChIP, ATAC, methylation) or `label` (binary STARR-seq annotation used as the training label) |


### Command

python3 Enhancer_annotation.py preprocessing \
  --output_path /path/to/output_dir \
  --input_file /path/to/Pre-processing_parameters.csv \
  --chrom_sizes /path/to/hg38.chrom.sizes.txt \
  --normalisation min_max \
  --binsize 100

### Parameters

| Flag               | Required | Default                  | Description                                                              |
|---------------------|:--------:|--------------------------|----------------------------------------------------------------------------|
| `--output_path`     | Yes      | path                  | Directory where output CSVs will be written.                              |
| `--input_file`       | Yes      | path                 | Features CSV described above (`Feature`, `path`, `type`).                  |
| `--chrom_sizes`      | No       | `hg38.chrom.sizes.txt`    | Path to the chromosome sizes file.                                        |
| `--normalisation`    | No       | `min_max`                 | Normalisation method: `min_max`, `quantile`, or `none`.                    |
| `--binsize`          | No       | `100`                      | Genome tile bin size, in bp.                                              |

### Output

- Feature matrix CSVs (tiled, normalised, and NA-masked versions), tiled by
  100bp bins across the genome.
- `FeatureMatrix_input_500k.csv` (training set), `FeatureMatrix_holdout_500k.csv`
  (test set), and `FeatureMatrix_unseen_500k.csv` (unseen/leftover genome bins),
  produced automatically from the feature matrix.

---

## 2. model training

Trains a new model on your own data, or runs an existing trained model to
annotate enhancers on new data. Supports four models: Logistic Regression
(`LR`/`LG`), XGBoost (`XGB`), CNN (`CNN`), and Type-2 Fuzzy Logic System
(`FUZZY`).


### Training a model


python3 Enhancer_annotation.py model_training \
  --train_file /path/to/Train_input_NAs.csv \
  --test_file /path/to/Test_input_NAs.csv \
  --model_name LR \
  --output_dir /path/to/output_dir




### Parameters

| Flag                | Required                                             | Default | Description                                                                                     |
|----------------------|-------------------------------------------------------|---------|---------------------------------------------------------------------------------------------------|
| `--train_file`       | Only when `--mode Training`                              | &mdash; | Path to the training data CSV (output of Pre-processing).                                        |
| `--test_file`        | Yes                                                     | &mdash; | Path to the test/inference data CSV.                                                              |
| `--model_name`       | Yes                                                     | &mdash; | Model to run: `LR`/`LG` (Logistic Regression), `XGB` (XGBoost), `CNN`, or `FUZZY`.                 |
| `--model_file`       | Only when `--mode` isn't `Training` and model isn't FUZZY | &mdash; | Path to a saved model file.                                                                        |
| `--partition_file`   | Only when `--mode` isn't `Training` and model is FUZZY   | &mdash; | Path to the fuzzy partition file.                                                                 |
| `--rule_file`        | Only when `--mode` isn't `Training` and model is FUZZY   | &mdash; | Path to the fuzzy rule file.                                                                       |
| `--output_dir`       | Yes                                                     | &mdash; | Directory where outputs (trained model, predictions, ROC curves, reports) will be written.        |

---
## 4. model prediction

### Running inference with an existing LR/XGBoost/CNN model


python3 Enhancer_annotation.py model_prediction \
  --test_file /path/to/Test_DMR_method_input_NAs.csv \
  --model_name CNN \
  --model_file /path/to/cnn_model.pth \
  --output_dir /path/to/output_dir

### Running inference with the Fuzzy (Type-2 FLS) model

The Fuzzy model uses a partition file and a rule file instead of a single
model file:


python3 Enhancer_annotation.py model_prediction \
  --test_file /path/to/Test_DMR_method_input_NAs.csv \
  --model_name FUZZY \
  --partition_file /path/to/Features_partitions.txt \
  --rule_file /path/to/Features_Rules_file.txt \
  --output_dir /path/to/output_dir



## 4. Enhancer merging

Takes the annotated-enhancer predictions from Model Training (a CSV
containing a `Probabilities` column) and merges the 100bp bins back into
full-length enhancer regions.

No parameter CSV is needed for this step either &mdash; everything is passed
on the command line, and the confidence column is fixed to `Probabilities`.

### Command


python3 Enhancer_annotation.py enhancer_merging \
  --enhancer_annotation_csv /path/to/LR_predictions.csv \
  --output_path /path/to/output_dir \
  --threshold 0.8 \
  --gap 500 \
  --bin_size 100


### Parameters

| Flag                          | Required | Default | Description                                                               |
|---------------------------------|:--------:|---------|------------------------------------------------------------------------------|
| `--enhancer_annotation_csv`      | Yes      | path | Path to the annotated-enhancer CSV (output of Model Training), must contain a `Probabilities` column. |
| `--output_path`                  | Yes      | path | Directory where output files will be written.                             |
| `--threshold`                    | No       | `0.8`    | Probability threshold above which a bin is considered part of an enhancer.|
| `--gap`                           | No       | `500`    | Maximum gap (bp) allowed when merging nearby bins into one enhancer domain.|
| `--bin_size`                      | No       | `100`    | Bin size in bp (should match the bin size used in Pre-processing).        |

### Output

- `glist_<threshold>_<gap>bp.pkl` / `.csv`; intermediate domain lists.
- `Annotated_Merged_Enhancers_<gap>bp.tsv` ; final merged, full-length
  enhancer regions.

---

## Notes

- Logistic Regression and XGBoost are relatively fast; CNN is the slowest of
  the four models to train/run.
- On small sample data, each step typically completes in under 15 minutes on
  a local system.
- Subcommand names (`preprocessing`, `model_training`, `model_prediction`,`enhancer_merging`) are
  case-insensitive.
