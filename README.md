# Implementations for Embedding Quantization (Compress Word Embeddings)

Thank you for your interest on our paper.

I'm receieving mail basically everyday and happy to know many of you implemented the model correctly.

And more than one engineers in large IT companies have told me that our idea works in their products.

I'm glad to debug your code or have discussion with you.

Please do not hesitate to mail me for help.

### Todos

1. The quality of the tensorflow code will be improved VERY soon
2. pytorch and theano code will be updated
3. Write usage of the codes

Raphael Shu, 2018.3

### Example of usage

1. Download glove data

```
cd neuralcompressor
bash scripts/download_glove_data.sh
```

2. Quantize embedding matrix

```
python tensorflow/embed_quantize.py
```

To be done

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
