# Explainable-Deep-Autoencoder

## Preamble

Deep learning has performed well and led the third wave of artificial intelligence. Most of the current top-performing applications use deep learning, and the big hit AlphaGo uses deep learning. Modern machine learning methods have been extensively utilized in gene expression data pro-cessing. In particular, autoencoders (AE) have been employed in processing noisy and hetero-genous RNA-Seq data. However, AEs usually lead to “black-box” hidden variables difficult to inter-pret, hindering downstream experimental validations and clinical translation. To bridge the gap be-tween complicated models and the biological interpretations, we developed a tool, XAE4Exp (eX-plainable AutoEncoder for Expression data), which integrates AE and SHapley Additive exPlana-tions (SHAP), a flagship technique in the field of eXplainable AI (XAI). It quantitatively evaluates the contributions of each gene to the hidden structure learned by an AE, substantially improving the expandability of AE outcomes. This tool will enable researchers and practitioners to analyze high-dimensional expression data intuitively, paving the way towards broader uses of deep learning.


---

## Table of Contents
- [Requirement](#requirement)
- [AutoEncoder and SHAP](#autoencoder-and-shap)
- [Prerequisites](#prerequisites)
- [Procedure](#procedure)
- [License](#license)
- [Authors](#author-info)
---
## Requirement
- python3.8 -m venv XAI_venv
- source XAI_venv/bin/activate
- pip install -r requirements.txt

## AutoEncoder and SHAP

**AutoEncoder**

Autoencoder is an unsupervised neural network model that learns the implicit features of the input data, here we use Deep AutoEncoder for representation learning.

**SHAP**

SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their related extensions. By using SHAP, we can track contribution (weights) of each gene from cancer gene for the resluts of representation learning.

For more details, please reach out to:

https://github.com/slundberg/shap#citations

## Prerequisites

- **Required Software**
  - Python 3.8.3 and above
  - R 1.4.0 and above

- **Python Libraries**
  - numpy
  - pandas
  - shap
  - sklearn
  - torch
  - matplotlib

- **R Libraries**
  - WebGestaltR

## Procedure

### 0. Data acquisition

- To demonstrate the use of XAE4Exp, we applied it to The Cancer Genome Atlas (TCGA) breast cancer data (sample size N= 1041, number of genes M= 56497). I would like to thank TCGA for their data support!

### 1. Data preparation

- (optional) To make the following feature importance figure more clear, we converted the data column of gene id to column of gene name by gene_annotation.R.

- According to **data quality control.py**. Before runing **AutoEncoder.py** (This script already contains data quality control coding, so basically you can just ran **AutoEncoder.py**), we need to do gene data quality control to make sure the the variance of each gene is less than 1, which denoises the gene data.

### 2. Representation learning by Deep AutoEncoder

- According to **AutoEncoder.py** (This script already contains data quality control). After gene quality control, we proceed to representation learning. By tuning the parameters in AutoEncoder.py, we would have representations in different precision level. In this Deep AE, a three-hidden-layer (exclude input and output layer) structure is used, and loss function is designed by Mean Square Error (MSE).

- Because we don't have a certain rule to tuning parameter, we always compare the results and find the best combination. Following is the parameter that would affect the results:
  - batch_size (number of data cloumns that computer runs each time)
  - test_size (train test split rate)
  - smallest_layer (the number that you want to compress gene data, by changing smallest_layer, you need to change the compress rate inside the **AutoEncoder Neural Network** correspondly.)
  - number_epochs (the number of round that you want data to be ran)

**<ins>Besides, you might change the number of hidden layer, which is so-called the depth of AutoEncoder.</ins>**

### 3. SHAP explanation

- According to **SHAP_combo.py**. SHAP is a strong deep learning explainer, by SHAP, you are able to get figures of distribution of each gene contribution and exact number of each gene contribution. Based on the flow of SHAP, a pre-designed prediction model and an explainer are need. Here we used a randon forest with 100 decision trees with maximun depth of 20 as the prediction model, and TreeExplainer as the explainer.

- The **SHAP_combo.py** results contains all feature importance in txt form (two columns with first column of gene name and second column of feature importance), top 20 global interpretation figure and top 20 feature importance figure. Separately, **SHAP_value.py** generates feature importance in txt file, **SHAP_figure_scatter.py** generates top 20 global interpretation figure, **SHAP_figure_bar.py** generates top 20 feature importance figure.

- Basically, you can just run **SHAP_combo.py**, or you can run **SHAP_value.py**, **SHAP_figure_scatter.py** and **SHAP_figure_bar.py** one-by-one seperately.

- SHAP figures:
  - Example for Global interpretation figure (**SHAP_figure_scatter.py**):
    <img width="613" alt="scatter6" src="https://user-images.githubusercontent.com/81887269/127747163-d6a1765c-b9b3-4313-ae0f-62d2a6f08327.png">
  - Example for Feature importance figure (**SHAP_figure_bar.py**):
    <img width="679" alt="bar6" src="https://user-images.githubusercontent.com/81887269/127747164-757099b9-8e23-4cd6-8755-9e83080cb8f2.png">

### 4. Enrichment Analysis

- According to **ORA.R** and **GSEA.R**. The enrichment analysis is preceeded by WebGestaltR package in R. ORA stands for Over-Representation Analysis and GSEA stands for Gene Set Enrichment Analysis. A **.csv** should be generated in this step.

- (optional) WebGestalt also provides online Enrichment Analysis service with clear figures. They provide Over-Representation Analysis (ORA), Gene Set Enrichment Analysis (GSEA) and Network Topology-based Analysis (NTA). Here is the website: http://www.webgestalt.org/#

### 5. Demonstration

- Example data: please find the example data at **data_example.csv**. Please note that the example data is fake and small size, the training result could be bad.
- If you are confused about the process, feel free to check out the demonstration in the folder **example**.


## License
**This project is licensed under the MIT License**

MIT License

Copyright (c) 2021 Yang Yu, Qingrun Zhang, Wenyuan Liao, Pathum Kossinna & Qing Li.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Author Info

- Yang Yu, Qingrun Zhang, Wenyuan Liao, Pathum Kossinna & Qing Li.
- Qingrun Zhang 
  - Email: qingrun.zhang@ucalgary.ca
- Yang Yu
  - Email: yang.yu2@ucalgary.ca
  

[Back To The Top](#table-of-contents)
