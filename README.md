Project 1: OpenCLIP ViT-B-32 Image Encoder (NumPy)
This repository contains my implementation of the OpenCLIP ViT-B-32 image encoder forward calculation, using pure NumPy. The implementation matches the reference outputs perfectly within the required tolerances.

Environment Setup
The project requires Python 3.9 or later. Dependencies can be installed via pip:

pip install -r requirements.txt

How to Run and Verify
To reproduce the results and run the numerical checks, execute the following commands in the project directory:

Verify the environment and assets:
python check.py --stage assets

Run all numerical checks:
python check.py --stage all

Generate the output tokens for the provided images:
python run.py --output output_tokens.npz

Implementation Details
All 6 required TODOs in image_encoder.py have been completed:

TODO-01 (patchify): Used reshape and transpose to split images from [B, 3, 224, 224] into 49 patches of [B, 49, 3072].

TODO-02 (split_heads): Reorganized the feature dimension into 12 attention heads, changing shape from [B, N, 768] to [B, 12, N, 64].

TODO-03 (MultiHeadAttention.call): Computed scaled dot-product scores, applied attention weights to values, merged the heads, and applied the final output projection.

TODO-04 (TransformerBlock.call): Implemented the MLP layer using the provided GELU activation and added both residual connections.

TODO-05 (embed): Applied the patch projection (matrix multiplication without bias), prepended the CLS token, and added learned positional embeddings.

TODO-06 (ImageEncoder.call): Selected the CLS token, applied post-LayerNorm, projected to 512 dimensions, and returned the unit-normalized output.

Numerical Accuracy
All stages pass the strict atol and rtol requirements. The final output token max absolute error is 5.439e-07, well within the 2e-05 tolerance.
