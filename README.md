# Sparsemax

Implementation of the Sparsemax activation function in Pytorch from the paper:  
[From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification](https://arxiv.org/abs/1602.02068) by André F. T. Martins and Ramón Fernandez Astudillo

This is a Pytorch port of https://github.com/gokceneraslan/SparseMax.torch/  
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
