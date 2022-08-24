# BottleMod: Modeling Data Flows and Tasks for Fast Bottleneck Analysis

This repository currently contains the first version of the BottleMod framework and scripts to generate the figures used in the paper with it.
This version is considered merely a proof of concept. A cleaner rewrite using only linear functions and putting more emphasis on performance is currently work in progress.

This first version does not support data output functions (section 2.4 in the paper). Instead the progress function *is* the tasks data output. Using data output functions the same can be achieved by having only one data output function O(p) = p.
The first version does depend on a modified version of SciPy. See the custom_scipy directory or dockerfile for more information.

Using docker:

```
# building the image may take several minutes
docker build -t bottlemod .

# running the image, creating the figures in a folder called "figures"
docker run --rm --mount type=bind,source="$(pwd)"/figures,target=/bottlemod/figures bottlemod
```
