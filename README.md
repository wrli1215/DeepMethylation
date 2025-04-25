# DeepMethylation
DeepMethylation consists of three main modules: a sequence module, an epigenomic module, and an integration module. The model takes a one-hot encoded DNA sequence around the CpG site and 25 epigenomic features annotated for the CpG site as input, and the binary DNA methylation state of CpG site centered in the DNA sequences as output. 

# Data collection and preprocessing
The Illumina EPIC Methylation array data for nine tissues (breast, kidney, colon, lung, muscle, ovary, prostate, testis, and whole blood) were obtained from the GTEx project, as published in the study conducted by Oliva et al. The dataset generated DNA methylation profiles of 754,119 CpG sites for 987 samples from GTEx v8 cohort. We aligned CpG sites detected across all samples from all tissue types and retained only commonly detected sites with comprehensive annotations. Genome sequences were derived from the UCSC hg19, GRCh37 (Genome Reference Consortium Human Reference 37) with GenBank assembly accession number GCA_000001405.1.

# Model training and evaluation
We split the CpGs into three independent sets by chromosome for model training and evaluation. Specifically, we used CpG sites on chromosomes 1 to 11 as the training set, comprising a total of 474,851 CpG sites. For model tuning, we used CpG sites on chromosomes 12 to 16 as the validation set, which included 144,234 CpG sites. The remaining CpG sites on chromosomes 17 to 22 constituted the testing set, consisting of 135,034 CpG sites, and were used for evaluating prediction performance. DeepMethylation was trained independently for each sample, with the final model evaluation results derived by averaging the outcomes across samples of the same tissue. We quantified the performance of the results using the accuracy (ACC), specificity (SP), sensitivity (SE), precision, and F1 score. Note that the functional CpG sites were those that were hypomethylated.
```
python DeepMethylation.py &
```

# Downstream analysis
Demo data were displayed in the folder "demo".  
* External validation on WGBS data.
```
python train850k_predict_WGBS.py
```
* Cross-platform imputation of DeepMethylation from 450k to EPIC array data.
```
python train450k_predict_850k.py
```
* Tissue-specific and cross-tissue generalization of DeepMethylation.
```
python CrossTissuePrediction.py
```
* Feature importance analysis.
```
python SHAP_ImportanceAnalysis.py
```

# Delta Methylation (DDM) model
We designed a variant-evaluation model, DDM, to predict the genetic effect of SNPs on the DNA methylation level of CpG sites.
```
python train450k_predict_850k.py
```

# Installation
Download DeepMethylation by
```
git clone https://github.com/wrli1215/DeepMethylation
```

# License
This project is licensed under the MIT License - see the LICENSE.md file for details.
