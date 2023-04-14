# Attention-Likelihood relationship in Transformers

<p align="center">
    <a href="https://github.com/grok-ai/nn-template"><img alt="NN Template" src="https://shields.io/badge/nn--template-0.2.3-emerald?style=flat&labelColor=gray"></a>
    <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.10-blue.svg"></a>
    <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

Code for the [ICLR 2023 TinyPapers](https://openreview.net/forum?id=R82eeIF4rP_) paper "Attention-likelihood relationship in Transformers":

```bibtex
@article{ruscio2023attention,
  title={Attention-likelihood relationship in transformers},
  author={Ruscio, Valeria and Maiorca, Valentino and Silvestri, Fabrizio},
  journal={arXiv preprint arXiv:2303.08288},
  year={2023}
}
```

![Alt text](correlation.png?raw=true "Attention-Likelihood correlation")

## Installation

```bash
pip install git+ssh://git@github.com/Flegyas/AttentionLikelihood.git
```


## Quickstart
The results are fully reproducible and the code is self-contained. There are 3 main files:
-  `utils.py` contains shared utility functions;
- `encode_likelihood_attention.ipynb` contains the code to extract the data from WordNet and encode it;
- `analyze_likelihood_attention.ipynb` contains the code to analyze the stored encodings, reproducing the tables and plots of the paper;


## Development installation

Setup the development environment:

```bash
git clone git@github.com:Flegyas/AttentionLikelihood.git
cd AttentionLikelihood
conda env create -f env.yaml
conda activate attlike
pre-commit install
```

### Update the dependencies

Re-install the project in edit mode:

```bash
pip install -e .[dev]
```
