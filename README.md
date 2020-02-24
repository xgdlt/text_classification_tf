## Support tasks

* Binary-class text classifcation
* Multi-class text classification
* Multi-label text classification
* Hiearchical (multi-label) text classification (HMC)

## Support text encoders

* TextCNN ([Kim, 2014](https://arxiv.org/pdf/1408.5882.pdf))
* RCNN ([Lai et al., 2015](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9745/9552))
* TextRNN ([Liu et al., 2016](https://arxiv.org/pdf/1605.05101.pdf))
* FastText ([Joulin et al., 2016](https://arxiv.org/pdf/1607.01759.pdf))
* VDCNN ([Conneau et al., 2016](https://arxiv.org/pdf/1606.01781.pdf))
* DPCNN ([Johnson and Zhang, 2017](https://www.aclweb.org/anthology/P17-1052))
* AttentiveConvNet ([Yin and Schutze, 2017](https://arxiv.org/pdf/1710.00519.pdf))
* DRNN ([Wang, 2018](https://www.aclweb.org/anthology/P18-1215))
* Region embedding ([Qiao et al., 2018](http://research.baidu.com/Public/uploads/5acc1e230d179.pdf))
* Transformer encoder ([Vaswani et al., 2017](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf))
* Star-Transformer encoder ([Guo et al., 2019](https://arxiv.org/pdf/1902.09113.pdf))

## Requirement

* Python 3
* Tensorflow 2.0+
* Numpy 1.14.3+


## Usage

### Training

    python train.py conf/train.json

***Detail configurations and explanations see [Configuration](readme/Configuration.md).***

The training info will be outputted in standard output and log.logger\_file.

### Evaluation
    python eval.py conf/train.json

* if eval.is\_flat = false, hierarchical evaluation will be outputted.
* eval.model\_dir is the model to evaluate.
* data.test\_json\_files is the input text file to evaluate.

The evaluation info will be outputed in eval.dir.

## Input Data Format

    JSON example:

    {
        "doc_label": ["Computer--MachineLearning--DeepLearning", "Neuro--ComputationalNeuro"],
        "doc_token": ["I", "love", "deep", "learning"],
        "doc_keyword": ["deep learning"],
        "doc_topic": ["AI", "Machine learning"]
    }

    "doc_keyword" and "doc_topic" are optional.

## Performance

### 0. Dataset

<table>
<tr><th>Dataset<th>Taxonomy<th>#Label<th>#Training<th>#Test
<tr><td>RCV1<td>Tree<td>103<td>23,149<td>781,265
<tr><td>Yelp<td>DAG<td>539<td>87,375<td>37,265
</table>

* RCV1: [Lewis et al., 2004](http://www.jmlr.org/papers/volume5/lewis04a/lewis04a.pdf)
* Yelp: [Yelp](https://www.yelp.com/dataset/challenge)

### 1. Compare with state-of-the-art
<table>
<tr><th>Text Encoders<th>Micro-F1 on RCV1<th>Micro-F1 on Yelp
<tr><td>HR-DGCNN (Peng et al., 2018)<td>0.7610<td>-
<tr><td>HMCN (Wehrmann et al., 2018)<td>0.8080<td>0.6640
<tr><td>Ours<td><strong>0.8313</strong><td><strong>0.6704</strong>
</table>

* HR-DGCNN: [Peng et al., 2018](http://www.cse.ust.hk/~yqsong/papers/2018-WWW-Text-GraphCNN.pdf)
* HMCN: [Wehrmann et al., 2018](http://proceedings.mlr.press/v80/wehrmann18a/wehrmann18a.pdf)


## Acknowledgement

Some public codes are referenced by our toolkit:


* https://github.com/ailias/Focal-Loss-implement-on-Tensorflow/
* https://github.com/brightmart/text_classification

## Update

* 2019-04-29, init version
