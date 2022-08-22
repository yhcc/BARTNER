This is the code for ACL-ICJNLP2021 paper [A Unified Generative Framework for Various NER Subtasks](https://arxiv.org/abs/2106.01223).

Install the package in the requirements.txt, then use the following
commands to install two other packages
```text
pip install git+https://github.com/fastnlp/fastNLP@dev
pip install git+https://github.com/fastnlp/fitlog
```

You need to put your data in the parallel folder of this repo
```text
    - BARTNER/
        - train.py
        ...
    - data/
        - conll2003
            - train.txt
            - text.txt
            - dev.txt
        - en-ontonotes
            - ...
        - Share_2013
        - Share_2014
        - CADEC
        - en_ace04
        - en_ace05
        - genia

```
For the `conll2003` and `en-ontonotes` you data in each split should like (The first column is words, the second column is tags. We assume the tag is the BIO-tagging)
```text
LONDON B-LOC
1996-08-30 O

West B-MISC
Indian I-MISC
all-rounder O
Phil B-PER
```

For nested dataset `en_ace04`, `en_ace05` and `genia`, the data should like 
(each line is a jsonline, contains ``ners`` and ``sentences`` keys.)
```text
{"ners": [[[16, 16, "DNA"], [4, 8, "DNA"], [24, 26, "DNA"], [19, 20, "DNA"]], [[31, 31, "DNA"], [2, 2, "DNA"], [4, 4, "DNA"], [30, 31, "DNA"]], [[23, 24, "RNA"], [14, 15, "cell_type"], [1, 2, "RNA"]], [[2, 2, "DNA"]], [], [[0, 0, "DNA"], [9, 9, "cell_type"]]], "sentences": [["There", "is", "a", "single", "methionine", "codon-initiated", "open", "reading", "frame", "of", "1,458", "nt", "in", "frame", "with", "a", "homeobox", "and", "a", "CAX", "repeat", ",", "and", "the", "open", "reading", "frame", "is", "predicted", "to", "encode", "a", "protein", "of", "51,659", "daltons."], ["When", "the", "homeodomain", "from", "HB24", "was", "compared", "to", "known", "mammalian", "and", "Drosophila", "homeodomains", "it", "was", "found", "to", "be", "only", "moderately", "conserved,", "but", "when", "it", "was", "compared", "to", "a", "highly", "diverged", "Drosophila", "homeodomain", ",", "H2.0,", "it", "was", "found", "to", "be", "80%", "identical."], ["The", "HB24", "mRNA", "was", "absent", "or", "present", "at", "low", "levels", "in", "normal", "B", "and", "T", "lymphocytes", ";", "however,", "with", "the", "appropriate", "activation", "signal", "HB24", "mRNA", "was", "induced", "within", "several", "hours", "even", "in", "the", "presence", "of", "cycloheximide", "."], ["Characterization", "of", "HB24", "expression", "in", "lymphoid", "and", "select", "developing", "tissues", "was", "performed", "by", "in", "situ", "hybridization", "."], ["Positive", "hybridization", "was", "found", "in", "thymus", ",", "tonsil", ",", "bone", "marrow", ",", "developing", "vessels", ",", "and", "in", "fetal", "brain", "."], ["HB24", "is", "likely", "to", "have", "an", "important", "role", "in", "lymphocytes", "as", "well", "as", "in", "certain", "developing", "tissues", "."]]}
{"ners": [[[16, 16, "DNA"], [4, 8, "DNA"], [24, 26, "DNA"], [19, 20, "DNA"]], [[31, 31, "DNA"], [2, 2, "DNA"], [4, 4, "DNA"], [30, 31, "DNA"]], [[23, 24, "RNA"], [14, 15, "cell_type"], [1, 2, "RNA"]], [[2, 2, "DNA"]], [], [[0, 0, "DNA"], [9, 9, "cell_type"]]], "sentences": [["There", "is", "a", "single", "methionine", "codon-initiated", "open", "reading", "frame", "of", "1,458", "nt", "in", "frame", "with", "a", "homeobox", "and", "a", "CAX", "repeat", ",", "and", "the", "open", "reading", "frame", "is", "predicted", "to", "encode", "a", "protein", "of", "51,659", "daltons."], ["When", "the", "homeodomain", "from", "HB24", "was", "compared", "to", "known", "mammalian", "and", "Drosophila", "homeodomains", "it", "was", "found", "to", "be", "only", "moderately", "conserved,", "but", "when", "it", "was", "compared", "to", "a", "highly", "diverged", "Drosophila", "homeodomain", ",", "H2.0,", "it", "was", "found", "to", "be", "80%", "identical."], ["The", "HB24", "mRNA", "was", "absent", "or", "present", "at", "low", "levels", "in", "normal", "B", "and", "T", "lymphocytes", ";", "however,", "with", "the", "appropriate", "activation", "signal", "HB24", "mRNA", "was", "induced", "within", "several", "hours", "even", "in", "the", "presence", "of", "cycloheximide", "."], ["Characterization", "of", "HB24", "expression", "in", "lymphoid", "and", "select", "developing", "tissues", "was", "performed", "by", "in", "situ", "hybridization", "."], ["Positive", "hybridization", "was", "found", "in", "thymus", ",", "tonsil", ",", "bone", "marrow", ",", "developing", "vessels", ",", "and", "in", "fetal", "brain", "."], ["HB24", "is", "likely", "to", "have", "an", "important", "role", "in", "lymphocytes", "as", "well", "as", "in", "certain", "developing", "tissues", "."]]}
...
```

For discontinuous dataset `Share_2013`, `Share_2014` and `CADEC`, the data should like (
each sample has two lines, if the second line is empty means there is not entity.
)
```text
Abdominal cramps , flatulence , gas , bloating .
0,1 ADR|3,3 ADR|7,7 ADR|5,5 ADR

Cramps would start within 15 minutes of taking pill , even during meals .
0,0 ADR

...
```
We use code from https://github.com/daixiangau/acl2020-transition-discontinuous-ner to pre-process
 the data.

You can run the code by directly using
```shell
python train.py
```

The following output should be achieved
```text
Save cache to caches/data_facebook/bart-large_conll2003_word.pt.                                                                                                        
max_len_a:0.6, max_len:10
In total 3 datasets:
        test has 3453 instances.
        train has 14041 instances.
        dev has 3250 instances.

The number of tokens in tokenizer  50265
50269 50274
input fields after batch(if batch size is 2):
        tgt_tokens: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2, 8]) 
        src_tokens: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2, 11]) 
        first: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2, 11]) 
        src_seq_len: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2]) 
        tgt_seq_len: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2]) 
target fields after batch(if batch size is 2):
        entities: (1)type:numpy.ndarray (2)dtype:object, (3)shape:(2,) 
        tgt_tokens: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2, 8]) 
        target_span: (1)type:numpy.ndarray (2)dtype:object, (3)shape:(2,) 
        tgt_seq_len: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2]) 

training epochs started 2021-06-02-11-49-26-964889
Epoch 1/30:   0%|                                                         | 15/32430 [00:06<3:12:37,  2.80it/s, loss:6.96158
```

Some important python files are listed below
```text
- BartNER
  - data
     - pipe.py # load and process data
  - model
     - bart.py # the model file
  - train.py  # the training file
```

The different ``Loader``s  in the `data/pipe.py` is meant to load data, and the ``data.BartNERPipe`` class 
is to process data, the loader should load data into a DataBundle object,
you can mock the provided Loader to write your own loader, as long as your
dataset has the following four fields, the ``BartNERPipe`` should be able to 
process it
```text
- raw_words  # List[str]
    # ['AL-AIN', ',', 'United', 'Arab', 'Emirates', '1996-12-06']
- entities  # List[List[str]]
    # [['AL-AIN'], ['United', 'Arab', 'Emirates']]
- entity_tags  # List[str], the same length as entities
    # ['loc', 'loc']
- entity_spans # List[List[int]], the inner list must have an even number of ints, means the start(inclusive，闭区间) and end(exclusive，开区间) of an entity segment
    # [[0, 1], [2, 5]] or for discontinous NER [[0, 1, 5, 7], [2, 3, 5, 7],...]
```

In order to help you reproduce the results, we have hardcoded the hyper-parameters
 for each dataset in the code, you can change them based on your need. 
We conduct all experiments in NVIDIA-3090(24G memory). Some known
 difficulties about the reproduction of this code: (1) Some datasets
(nested and discontinous) will drop to 0 or near 0 F1 during training, please drop these
 results; (2) randomness will cause large performance variance for some datasets, please try to 
run multiple times. 

We deeply understand how frustrating it can be 
if the results are hard to reproduce, we tried our best to make sure 
the results were at least reproducible in our equipment (Usually take 
average from at least  five runs).


### Some questions asked by others
#### 1. Where to get the metric?  
Since the evaluation takes several seconds, the code will start 
   to evaluate after a certain epoch (based on our experiments, 
   the best performance almost always achieved after the pre-set
   eval_start_epoch). You can change this value (in train.py) to make it evaluate
   earlier. 

#### 2. About the split of the Conll2003 dataset.
We follow previous work to concat the train and dev set (which will merge automatically in the train.py) as the 
   train file. Therefore, the output performance of conll2003 is the test performance.

#### 3. About the split of the Genia dataset.
Some previous work had no dev split, but we split 10% of the training set to be as the dev set(
   which will split the train data automatically in the train.py, and the split is deterministic, therefore, follow 
   the code, you can get the same split as ours). 



