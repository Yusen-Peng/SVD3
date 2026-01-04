# Related Work

SVD-LLM V2:
- dynamic compression ratio:
    - ![alt text](docs/dynamic_ratio.png)
- two-step SVD instead of Cholesky decomposition
    - ![alt text](docs/different_svd.png)

AdaSVD:
- dynamic compression ratio:
    - ![alt text](docs/important_score.png)
- alternating updates (no supervised lora at all):
    - ![alt text](docs/alternating_updates.png)

DipSVD:
- dynamic compression ratio:
    - ![alt text](docs/fisher_erank.png)
- channel-weighted whitening:
    - ![alt text](docs/channel_aware.png)