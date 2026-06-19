# RUDAS: Rescaling Huge Datasets based on Age Score

**Abstract:**

* Efficiently rescaling a large dataset by filtering samples based on age scores accumulated during warm-up training of a lightweight backbone network.
* A unified collection of sensitivity-filtered images forming a compact yet challenging subset that preserves the relative difficulty of the full dataset.
* An application to rescaling two large datasets — ImageNet and Places365 — to obtain rescaled subsets at multiple retention ratios $r$.
* Experimental results for image classification validate the representativeness of the rescaled subsets across multiple lightweight architectures.
* Strong performance on a rescaled subset is indicative of strong performance on the full dataset, allowing researchers to save time and computational cost during early network development.

**Note**: The work is currently submitted to ACML 2026 for peer-review, as of 20/06/2026. 

---

## Rescaling Explanations

* **Rescaled subsets of ImageNet** — subsets $IN^{r}$ at retention ratios $r \in \{0.1, 0.2, 0.3, 0.4, 0.5\}$ stored under `Rescaled_ImageNet/`
* **Rescaled subsets of Places365** — subsets $PL^{r}$ at retention ratios $r \in \{0.1, 0.2, 0.3, 0.4, 0.5\}$ stored under `Rescaled_Places365/`

**Phase 1 — Compute age scores (warm-up training + scoring pass):**

```
$ python age_script.py --data /path/to/ImageNet --epochs 100 --output_dir age_scores
```

Age scores (`age_scores_NNN.npy`) are saved incrementally after each epoch to `age_scores/age_scoring/`. The run can be safely interrupted and resumed via `--resume`. Age scores of ImageNet and Places365 that we have measured can be accessed via this [link](https://drive.google.com/drive/folders/1SB_8EWoMEvM6JjcCPpS6lRnMNpRy18ZO?usp=sharing).


**Phase 2 — Prune datasets from the age table:**

From the `age_scores_NNN.npy`, run `age_table.py` to produce the csv file with indices as the relative path to each sample of the original dataset. This csv file is then used for running the pruning script. For example, to prune ImageNet with preservation rate `r=0.1`, each easy / hard collection accounts for `b=1`, we run the following:

```
$ python pruning_age.py -d path/to/ImageNet --age path/to/age_table.csv -re 0.1 -rm 0.1 -rh 0.1 -q 0.1 
```

Modify the script accordingly for Places365 or any other datasets.



**Phase 3 — Train classifiers on rescaled subsets of ImageNet and Places365:**

* For training CNN-backbones on rescaled sub-datasets of ImageNet

```
$ python train_Rescaled_ImageNet --data path/to/Rescaled_ImageNet 
```

* For training CNN-backbones on rescaled sub-datasets of Places365

```
$ python train_Rescaled_Places365 --data path/to/Rescaled_Places365 
```

---

## Experimental Results — Top-1 Accuracy (%) on Rescaled Subsets

Top-1 accuracy (%) across backbones on the $r=0.1$ rescaled subsets vs. full datasets:

| Network | IN$^{r=0.1}$ | ImageNet | PL$^{r=0.1}$ | Places365 |
|:---|---:|---:|---:|---:|
| GoogLeNet | 35.77 | 68.30 | 40.02 | 53.63 |
| ShuffleNetV1 | 42.27 | 67.80 | 47.15 | — |
| ShuffleNetV2 | 43.97 | 69.36 | 44.00 | 50.80 |
| MobileNetV1 | 46.65 | 70.60 | 44.90 | 53.50 |
| MobileNetV3 | 46.71 | 71.50 | 44.95 | 53.53 |
| MobileNetV2 | 46.95 | 72.00 | 46.64 | 52.19 |


MobileNetV1 on rescaled subsets $\overline{\mathcal{D}^r}$ of ImageNet and Places365:

| Rescaled subset | $r$ | $\overline{\mathcal{D}^r_{train}}$ images | $\overline{\mathcal{D}^r_{valid}}$ images | MobileNetV1 |
|:---|---:|---:|---:|---:|
| IN$^{r=0.1}$ | 0.1 | 123,998 | 12,000 | 46.65 |
| IN$^{r=0.2}$ | 0.2 | 248,528 | 12,000 | 57.09 |
| IN$^{r=0.3}$ | 0.3 | 381,860 | 12,000 | 62.64 |
| IN$^{r=0.4}$ | 0.4 | 506,460 | 14,000 | 62.61 |
| IN$^{r=0.5}$ | 0.5 | 640,085 | 24,000 | 66.50 |
| PL$^{r=0.1}$ | 0.1 | 180,200 | 4,380 | 44.90 |
| PL$^{r=0.2}$ | 0.2 | 360,576 | 5,110 | 48.20 |
| PL$^{r=0.3}$ | 0.3 | 540,893 | 9,490 | 49.99 |
| PL$^{r=0.4}$ | 0.4 | 721,266 | 13,870 | 51.48 |
| PL$^{r=0.5}$ | 0.5 | 901,656 | 18,250 | 52.15 |


---

## Related Citations

If you use any materials from this repository, please cite the following relevant works.

```bibtex
@article{prlNguyen23,
  author  = {Thanh Tuan Nguyen and Thanh Phuong Nguyen},
  title   = {Rescaling Large Datasets Based on Validation Outcomes of a Pre-trained Network},
  journal = {Pattern Recognition Letters},
  volume  = {185},
  pages   = {73--80},
  year    = {2024},
  url     = {https://doi.org/10.1016/j.patrec.2024.07.001},
}
```

```bibtex
@unpublished{amclNguyen26,
  author = {Nguyen, Trung Duc and Nguyen, Thanh Tuan and Borgi, Mohamed Anouar and Nguyen, Thanh Phuong},
  title  = {Rescaling Huge Datasets based on Age Score},
  note   = {Manuscript submitted for publication to \textit{Machine Learning} (ACML Journal Track)},
  year   = {2026},
}
```
