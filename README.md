# Explainable-Deep-Autoencoder

## Preamble

Deep neural networks are emerging tools in learning representations of high-dimensional biological data including gene expressions. However, they usually lead to “black-boxes” difficult to interpret, hindering downstream experimental validations and clinical translation. To bridge the gap between complicated models and the needs of biological researchers, we developed a tool integrating autoencoder (AE) and SHapley Additive exPlanations (SHAP), a flagship technique in explainable AI (XAI). It quantitatively evaluates the contributions of each gene to the hidden structure learned by AE, substantially improving the expandability of AE models. By applying our tool to gene expression data, we revealed intriguing pathways including DNA reparation underlying breast cancer. This tool will enable researchers and practitioners to analyze high-dimensional genomic data intuitively, paving the way towards practical use of deep learning in broader biological and clinical applications.

---

## Table of Contents

- [AutoEncoder and SHAP](#autoencoder-and-shap)
- [Prerequisites](#Prerequisites)
- [Procedure](#procedure)
- [License](#license)
- [Authors](#author-info)
- [Citations](#citations)
---
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

- **Python Libraries**
  - numpy
  - pandas
  - pickle
  - shap
  - sklearn
  - torch
  - matplotlib

## Procedure

### 1.Data preparation

- According to **data quality control.py**. Before runing **AutoEncoder.py**, we need to do gene data quality control to make sure the the variance of each gene is less than 1, which denoise the gene data.

### 2.Representation learning by Deep AutoEncoder

- According to **AutoEncoder.py** (This script already contains data quality control). After gene quality control, we proceed to representation learning. By tuning the parameters in AutoEncoder.py, we would have representations in different precision level. Because we don't have a certain rule to tuning parameter, we always compare the results and find the best combination. Following is the parameter that would affect the results:
  - batch_size (number of data cloumns that computer runs each time)
  - test_size (train test split rate)
  - smallest_layer (the number that you want to compress gene data, by changing smallest_layer, you need to change the compress rate inside the **AutoEncoder Neural Network** correspondly.)
  - number_epochs (the number of round that you want data to be ran)

**<ins>Besides, you might change the number of hidden layer, which is so-called the depth of AutoEncoder.</ins>**

### 3.SHAP explanation

- SHAP is a strong deep learning explainer, by SHAP, you are able to get figures of distribution of each gene contribution and exact number of each gene contribution. 
- For figures, please run **SHAP results as figures.py**
  - Example figure
    [shap_BRCA.pdf](https://github.com/yancy001/Explainable-Deep-Autoencoder/files/6283423/shap_BRCA.pdf)

- For weights, please run **SHAP results for AE results.py**

### 4.（Optional) SHAP value (weights) filtering

- According to **threshold for SHAP results.py.** Before Enrichment Analysis, too many SHAP value might generate a huge amount of related pathway and it could be less accurate. To solve this problem, setup a threshold to filter the top gene id would be helpful. It is important to note that if the threshold you setup is too large, the gene number might be not enough to do Enrichment Analysis.

### 5.Enrichment Analysis

- WebGestalt provides online Enrichment Analysis service with clear figures. They provide Over-Representation Analysis (ORA), Gene Set Enrichment Analysis (GSEA) and Network Topology-based Analysis (NTA). Here is the website: http://www.webgestalt.org/#
- Example for Enrichment Analysis figure:

  <img width="1192" alt="Screen Shot 2021-04-09 at 12 11 21 AM" src="https://user-images.githubusercontent.com/81887269/114136557-2796af80-98c8-11eb-9bd7-11c77d7abbf7.png">


## License
**This project is licensed under the MIT License**

MIT License

Copyright (c) 2021 Yang Yu

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

- Yang Yu, Pathum Kossinna, Qing Li, Wenyuan Liao & Qingrun Zhang.

[Back To The Top](#table-of-contents)
