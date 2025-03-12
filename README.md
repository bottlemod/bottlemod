# BottleMod: Modeling Data Flows and Tasks for Fast Bottleneck Analysis

This repository contains the software to the BottleMod paper published at ICPE'25. For citation, please use the following bibtex:

```
@inproceedings{10.1145/3676151.3719382,
author = {Ansgar L{\"{o}}{\ss}er and
          Joel Witzke and
          Florian Schintke and
          Bj{\"{o}}rn Scheuermann},
title = {{BottleMod}: Modeling Data Flows and Tasks for Fast Bottleneck Analysis},
year = {2025},
isbn = {979-8-4007-1073-5/2025/05},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3676151.3719382},
doi = {10.1145/3676151.3719382},
booktitle = {Proceedings of the 16th ACM/SPEC International Conference on Performance Engineering},
numpages = {8},
keywords = {model, scientific workflows, data analysis workflows, bottleneck analysis},
location = {Toronto, Canada},
series = {ICPE '25}
}
```

BottleMod uses a slightly modified version of SciPy making the function that describes a piece of a piecewise function not begin from their the piece's start but from absolute 0. For ease of use, we provide a dockerfile to setup the necessary dependencies.
Executing the image will generate the base figures as used in the paper. They are not the very same figures, as they got extended with TikZ for the paper.

Usage (especially building the image may take a while):

```bash
# build the image
docker build -t bottlemod .

# execute it, saving the generated figures to the folder /path/to/figures
docker run --rm -v /path/to/figures:/bottlemod/figures bottlemod
```

BottleMod itself is implemented in `func.py`, `ppoly.py` and `task.py`.
`paper_figures_eval.py` and `paper_figures_general.py` make use of BottleMod to calculate and draw the figures used in the paper. They therefore serve as usage examples. That includes the code being executed, but also the code commented out. Those were other examples and plots executed at some point but not used in the final version of the paper.
While `paper_figures_general.py` uses synthetic examples, `paper_figures_eval.py` depicts the evaluation case for the simple video processing workflow described in the paper.
