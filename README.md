# Introduction
The PyTorch implementation for the following paper:
## Fully Unsupervised Deepfake Video Detection via Enhanced Contrastive Learning
![alt text](Pipeline.png "Illustration of the proposed fully unsupervised framework")
Paper link: (https://ieeexplore.ieee.org/abstract/document/10411047)

# Changelog
2024.1.22: Pre-released this repository.
2024.1.30: Updated the implementation in stage1.

# Environment
Python==3.7.6, torch==1.10.1, torchvision==0.11.2, cudatoolkit==11.1

# Run
Stage 1 code: ``pseudo_label_generator.py``

Stage 2 code: ``enhanced_contrastive_learner.py``

Stage 3 code: ``binary_classifier.py``

# Citations
Please cite the following paper in your publications if you use the python implementations:
```
@article{qiao2024fully,
  title={Fully Unsupervised Deepfake Video Detection via Enhanced Contrastive Learning},
  author={Qiao, Tong and Xie, Shichuang and Chen, Yanli and Retraint, Florent and Luo, Xiangyang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  publisher={IEEE}
}
```

# Acknowledgments
[SimCLR](https://github.com/sthalles/SimCLR)

[SupContrast](https://github.com/HobbitLong/SupContrast)

[EVA](https://github.com/FalkoMatern/Exploiting-Visual-Artifacts)
