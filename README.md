# BottleMod: Modeling Data Flows and Tasks for Fast Bottleneck Analysis

This repository contains the software to the BottleMod paper published at ICPE'25.
BottleMod uses a slightly modified version of SciPy. For ease of use, we provide a dockerfile to setup the necessary dependencies.
Executing the image will generate the base figures as used in the paper. They are not the very same figures, as they got extended with TikZ for the paper.

Usage (especially building the image may take a while):

```bash
# build the image
docker build -t bottlemod .

# execute it, saving the generated figures to the folder /path/to/figures
docker run --rm -v /path/to/figures:/bottlemod/figures bottlemod
```

BottleMod itself is implemented in func.py, ppoly.py and task.py.
paper_figures_eval.py and paper_figures_general.py make use of BottleMod to calculate and draw the figures used in the paper. They therefore serve as usage examples. That includes the code being executed, but also the code commented out. Those were other examples and plots executed at some point but not used in the final version of the paper.
