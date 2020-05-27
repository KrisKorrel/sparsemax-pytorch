# Sparsemax

Implementation of the Sparsemax activation function in Pytorch from the paper:  
[From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification](https://arxiv.org/abs/1602.02068) by André F. T. Martins and Ramón Fernandez Astudillo

Tested in Pytorch 0.4.0

Example usage
```python
import torch
from sparsemax import Sparsemax

sparsemax = Sparsemax(dim=1)
softmax = torch.nn.Softmax(dim=1)

logits = torch.randn(2, 5)
print("\nLogits")
print(logits)

softmax_probs = softmax(logits)
print("\nSoftmax probabilities")
print(softmax_probs)

sparsemax_probs = sparsemax(logits)
print("\nSparsemax probabilities")
print(sparsemax_probs)
```

Please add an issue if you have questions or suggestions.

# References
[From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification](https://arxiv.org/abs/1602.02068)

Note that is a Pytorch port of an existing implementation: https://github.com/gokceneraslan/SparseMax.torch/  

DOI for this particular repository: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3860669.svg)](https://doi.org/10.5281/zenodo.3860669)

This implementation was used for [Transcoding Compositionally: Using Attention to Find More Generalizable Solutions](https://arxiv.org/abs/1906.01234)
