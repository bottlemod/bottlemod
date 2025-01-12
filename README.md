# BottleMod: Modeling Data Flows and Tasks for Fast Bottleneck Analysis

This repository contains the software to the BottleMod paper published at ICPE'25.
BottleMod uses a slightly modified version of SciPy. For ease of use, we provide a dockerfile to setup the necessary dependencies.
Executing the image will generate the base figures as used in the paper. They are not the very same figures, as they got extended with TikZ for the paper.

Usage (especially building the image may take a while):

```bash
# build the image
docker build -t bottlemod -f dockerfile

# execute it, saving the generated figures to the folder /path/to/figures
docker run --rm -v /path/to/figures:/bottlemod/figures bottlemod
```
