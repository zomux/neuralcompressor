# nncompress: Implementations of Embedding Quantization (Compress Word Embeddings)

Thank you for your interest on our paper.

I'm receieving mail basically everyday and happy to know many of you implemented the model correctly.

I'm glad to debug your code or have discussion with you.

Please do not hesitate to mail me for help.

`mail_address = "raph_ael@ua_ca.com".replace("_", "")`

### Requirements:

numpy and tensorflow (I also have the pytorch implementation, which will be uploaded)

### Tutorial of the code

1. Download the project and prepare the data

```
> git clone https://github.com/zomux/neuralcompressor
> cd neuralcompressor
> bash scripts/download_glove_data.sh
```

2. Convert the Glove embeddings to numpy format

```
> python scripts/convert_glove2numpy.py data/glove.6B.300d.txt
```

3. Train the embedding quantization model

```
> python bin/quantize_embed.py -M 32 -K 16 --train
```

```
...
[epoch198] train_loss=12.82 train_maxp=0.98 valid_loss=12.50 valid_maxp=0.98 bps=618 *
[epoch199] train_loss=12.80 train_maxp=0.98 valid_loss=12.53 valid_maxp=0.98 bps=605
Training Done
```

4. Evaluate the averaged euclidean distance

```
> python bin/quantize_embed.py -M 32 -K 16 --evaluate
```

```
Mean euclidean distance: 4.889592628145218
```

5. Export the word codes and the codebook matrix

```
> python bin/quantize_embed.py -M 32 -K 16 --export
```

It will generate two files:
- data/mymodel.codes
- data/mymodel.codebook.npy

6. Check the codes

```
> paste data/glove.6B.300d.word data/mymodel.codes | head -n 100
```

```
...
only    15 14 7 10 1 14 14 3 0 9 1 9 3 3 0 0 12 1 3 12 15 3 11 12 12 6 1 5 13 6 2 6
state   7 13 7 3 8 14 10 6 6 4 12 2 9 3 9 0 1 1 3 9 11 10 0 14 14 4 15 5 0 6 2 1
million 5 7 3 15 1 14 4 0 6 11 1 4 8 3 1 0 0 1 3 14 8 6 6 5 2 1 2 12 13 6 6 15
could   3 14 7 0 2 14 5 3 0 9 1 0 2 3 9 0 3 1 3 11 5 15 1 12 12 6 1 6 2 6 2 10
...
```

### Use it in python

```python
from nncompress import EmbeddingCompressor

# Load my embedding matrix
matrix = np.load("data/glove.6B.300d.npy")

# Initialize the compressor
compressor = EmbeddingCompressor(32, 16, "data/mymodel")

# Train the quantization model
compressor.train(matrix)

# Evaluate
distance = compressor.evaluate(matrix)
print("Mean euclidean distance:", distance)

# Export the codes and codebook
compressor.export(matrix, "data/mymodel")
```

### Citation

```
@inproceedings{shu2018compressing,
title={Compressing Word Embeddings via Deep Compositional Code Learning},
author={Raphael Shu and Hideki Nakayama},
booktitle={International Conference on Learning Representations (ICLR)},
year={2018},
url={https://openreview.net/forum?id=BJRZzFlRb},
}
```

Arxiv version: https://arxiv.org/abs/1711.01068
