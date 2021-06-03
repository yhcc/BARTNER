from fastNLP.io import ConllLoader, Loader
from fastNLP.io.loader.conll import _read_conll
from fastNLP.io.pipe.utils import iob2, iob2bioes
from fastNLP import DataSet, Instance
from fastNLP.io import Pipe
from transformers import AutoTokenizer
from fastNLP.core.metrics import _bio_tag_to_spans
from fastNLP.io import DataBundle
import numpy as np
from itertools import chain
from fastNLP import Const
from functools import cmp_to_key
import json
from copy import deepcopy
from tqdm import tqdm


class BartNERPipe(Pipe):
    def __init__(self, tokenizer='facebook/bart-large', dataset_name='conll2003', target_type='word'):
        """

        :param tokenizer:
        :param dataset_name:
        :param target_type:
            支持word: 生成word的start;
            bpe: 生成所有的bpe
            span: 每一段按照start end生成
            span_bpe: 每一段都是start的所有bpe，end的所有bpe
        """
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        assert target_type in ('word', 'bpe', 'span')

        if dataset_name == 'conll2003':
            self.mapping = {
                'loc': '<<location>>',
                'per': '<<person>>',
                'org': '<<organization>>',
                'misc': '<<others>>',
            }  # 记录的是原始tag与转换后的tag的str的匹配关系
        elif dataset_name == 'en-ontonotes':
            self.mapping = \
                {'person': '<<person>>', 'gpe': '<<government>>', 'org': "<<organization>>",
                 'date': "<<date>>", 'cardinal': '<<cardinal>>', 'norp': "<<nationality>>", 'money': "<<money>>",
                 'percent': '<<percent>>', 'ordinal': '<<ordinal>>', 'loc': '<<location>>', 'time': "<<time>>",
                 'work_of_art': "<<work_of_art>>", 'fac': "<<buildings>>", 'event': "<<event>>",
                 'quantity': "<<quantity>>", 'product': "<<product>>", 'language': "<<language>>", 'law': "<<law>>"}
        elif dataset_name == 'en_ace04':
            self.mapping = {
                'loc': '<<location>>', "gpe": "<<government>>", "wea": "<<weapon>>", 'veh': "<<vehicle>>",
                'per': '<<person>>',
                'org': '<<organization>>',
                'fac': '<<buildings>>',
            }  # 记录的是原始tag与转换后的tag的str的匹配关系

        cur_num_tokens = self.tokenizer.vocab_size
        self.num_token_in_orig_tokenizer = cur_num_tokens
        self.target_type = target_type

    def add_tags_to_special_tokens(self, data_bundle):
        if not hasattr(self, 'mapping'):
            from collections import Counter
            counter = Counter()
            data_bundle.apply_field(counter.update, field_name='entity_tags', new_field_name=None)
            mapping = {}
            for key, value in counter.items():
                mapping[key] = '<<' + key + '>>'
            self.mapping = mapping
        else:
            mapping = self.mapping

        tokens_to_add = sorted(list(mapping.values()), key=lambda x: len(x), reverse=True)
        unique_no_split_tokens = self.tokenizer.unique_no_split_tokens
        sorted_add_tokens = sorted(list(tokens_to_add), key=lambda x: len(x), reverse=True)
        for tok in sorted_add_tokens:
            assert self.tokenizer.convert_tokens_to_ids([tok])[0] == self.tokenizer.unk_token_id
        self.tokenizer.unique_no_split_tokens = unique_no_split_tokens + sorted_add_tokens
        self.tokenizer.add_tokens(sorted_add_tokens)
        self.mapping2id = {}  # 给定转换后的tag，输出的是在tokenizer中的id，用来初始化表示
        self.mapping2targetid = {}  # 给定原始tag，输出对应的数字

        for key, value in self.mapping.items():
            key_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(value))
            assert len(key_id) == 1, value
            assert key_id[0] >= self.num_token_in_orig_tokenizer
            self.mapping2id[value] = key_id[0]  #
            self.mapping2targetid[key] = len(self.mapping2targetid)

    def process(self, data_bundle):
        """
        支持的DataSet的field为

            entities: List[List[str]], 每个元素是entity，非连续的拼到一起了
            entity_tags: 与上面一样长，是每个entity的tag
            raw_words: List[str]词语
            entity_spans： List[List[int]]记录的是上面entity的start和end，这里的长度一定是偶数，是start,end的pair, end是开区间

        :param data_bundle:
        :return:
        """
        self.add_tags_to_special_tokens(data_bundle)

        # 转换tag
        target_shift = len(self.mapping) + 2  # 是由于第一位是sos，紧接着是eos, 然后是

        def prepare_target(ins):
            raw_words = ins['raw_words']
            word_bpes = [[self.tokenizer.bos_token_id]]
            first = []  # 用来取每个word第一个bpe
            cur_bpe_len = 1
            for word in raw_words:
                bpes = self.tokenizer.tokenize(word, add_prefix_space=True)
                bpes = self.tokenizer.convert_tokens_to_ids(bpes)
                first.append(cur_bpe_len)
                cur_bpe_len += len(bpes)
                word_bpes.append(bpes)
            assert first[-1] + len(bpes) == sum(map(len, word_bpes))
            word_bpes.append([self.tokenizer.eos_token_id])
            assert len(first) == len(raw_words) == len(word_bpes) - 2

            lens = list(map(len, word_bpes))
            cum_lens = np.cumsum(lens).tolist()

            entity_spans = ins['entity_spans']  # [(s1, e1, s2, e2), ()]
            entity_tags = ins['entity_tags']  # [tag1, tag2...]
            entities = ins['entities']  # [[ent1, ent2,], [ent1, ent2]]
            target = [0]  # 特殊的sos
            pairs = []

            first = list(range(cum_lens[-1]))

            assert len(entity_spans) == len(entity_tags)
            _word_bpes = list(chain(*word_bpes))
            for idx, (entity, tag) in enumerate(zip(entity_spans, entity_tags)):
                cur_pair = []
                num_ent = len(entity) // 2
                for i in range(num_ent):
                    start = entity[2 * i]
                    end = entity[2 * i + 1]
                    cur_pair_ = []
                    if self.target_type == 'word':
                        cur_pair_.extend([cum_lens[k] for k in list(range(start, end))])
                    elif self.target_type == 'span':
                        cur_pair_.append(cum_lens[start])
                        cur_pair_.append(cum_lens[end]-1)  # it is more reasonable to use ``cur_pair_.append(cum_lens[end-1])``
                    elif self.target_type == 'span_bpe':
                        cur_pair_.extend(
                            list(range(cum_lens[start], cum_lens[start + 1])))  # 由于cum_lens是[1, 3...]即第0位其实就是cls之后的了
                        cur_pair_.extend(
                            list(range(cum_lens[end - 1], cum_lens[end])))  # 由于cum_lens是[1, 3...]即第0位其实就是cls之后的了
                    elif self.target_type == 'bpe':
                        cur_pair_.extend(list(range(cum_lens[start], cum_lens[end])))
                    else:
                        raise RuntimeError("Not support other tagging")
                    cur_pair.extend([p + target_shift for p in cur_pair_])
                for _, (j, word_idx) in enumerate(zip((cur_pair[0], cur_pair[-1]), (0, -1))):
                    j = j - target_shift
                    if 'word' == self.target_type or word_idx != -1:
                        assert _word_bpes[j] == \
                               self.tokenizer.convert_tokens_to_ids(
                                   self.tokenizer.tokenize(entities[idx][word_idx], add_prefix_space=True)[:1])[0]
                    else:
                        assert _word_bpes[j] == \
                               self.tokenizer.convert_tokens_to_ids(
                                   self.tokenizer.tokenize(entities[idx][word_idx], add_prefix_space=True)[-1:])[0]
                assert all([cur_pair[i] < cum_lens[-1] + target_shift for i in range(len(cur_pair))])

                cur_pair.append(self.mapping2targetid[tag] + 2)  # 加2是由于有shift
                pairs.append([p for p in cur_pair])
            target.extend(list(chain(*pairs)))
            target.append(1)  # 特殊的eos

            word_bpes = list(chain(*word_bpes))
            assert len(word_bpes)<500

            dict  = {'tgt_tokens': target, 'target_span': pairs, 'src_tokens': word_bpes,
                    'first': first}
            return dict

        data_bundle.apply_more(prepare_target, use_tqdm=True, tqdm_desc='pre. tgt.')

        data_bundle.set_ignore_type('target_span', 'entities')
        data_bundle.set_pad_val('tgt_tokens', 1)  # 设置为eos所在的id
        data_bundle.set_pad_val('src_tokens', self.tokenizer.pad_token_id)

        data_bundle.apply_field(lambda x: len(x), field_name='src_tokens', new_field_name='src_seq_len')
        data_bundle.apply_field(lambda x: len(x), field_name='tgt_tokens', new_field_name='tgt_seq_len')
        data_bundle.set_input('tgt_tokens', 'src_tokens', 'src_seq_len', 'tgt_seq_len', 'first')
        data_bundle.set_target('tgt_tokens', 'tgt_seq_len', 'target_span', 'entities')

        return data_bundle

    def process_from_file(self, paths, demo=False) -> DataBundle:
        """

        :param paths: 支持路径类型参见 :class:`fastNLP.io.loader.ConllLoader` 的load函数。
        :return: DataBundle
        """
        # 读取数据
        if isinstance(paths, str):
            path = paths
        else:
            path = paths['train']
        if 'conll2003' in path or 'ontonotes' in path:
            data_bundle = Conll2003NERLoader(demo=demo).load(paths)
        # elif 'ontonotes' in path:
        #     data_bundle = OntoNotesNERLoader(demo=demo).load(paths)
        elif 'genia' in path:
            data_bundle = NestedLoader(demo=demo).load(paths)
        elif 'en_ace0' in path:
            data_bundle = NestedLoader(demo=demo).load(paths)
        else:
            data_bundle = DiscontinuousNERLoader(demo=demo).load(paths)
        data_bundle = self.process(data_bundle)
        return data_bundle


class Conll2003NERLoader(ConllLoader):
    r"""
    用于读取conll2003任务的NER数据。每一行有4列内容，空行意味着隔开两个句子

    支持读取的内容如下
    Example::

        Nadim NNP B-NP B-PER
        Ladki NNP I-NP I-PER

        AL-AIN NNP B-NP B-LOC
        United NNP B-NP B-LOC
        Arab NNP I-NP I-LOC
        Emirates NNPS I-NP I-LOC
        1996-12-06 CD I-NP O
        ...

    返回的DataSet的内容为

        entities: List[List[str]], 每个元素是entity，非连续的拼到一起了
        entity_tags: 与上面一样长，是每个tag的分数
        raw_words: List[str]词语
        entity_spans： List[List[int]]记录的是上面entity的start和end，这里的长度一定是偶数，是start,end的pair, end是开区间

    """

    def __init__(self, demo=False):
        headers = [
            'raw_words', 'target',
        ]
        super().__init__(headers=headers, indexes=[0, 1])
        self.demo = demo

    def _load(self, path):
        r"""
        传入的一个文件路径，将该文件读入DataSet中，field由ConllLoader初始化时指定的headers决定。

        :param str path: 文件的路径
        :return: DataSet
        """
        ds = DataSet()
        for idx, data in _read_conll(path, indexes=self.indexes, dropna=self.dropna):
            doc_start = False
            for i, h in enumerate(self.headers):
                field = data[i]
                if str(field[0]).startswith('-DOCSTART-'):
                    doc_start = True
                    break
            if doc_start:
                continue
            ins = {h: data[i] for i, h in enumerate(self.headers)}
            raw_words = ins['raw_words']
            target = iob2(ins['target'])
            spans = _bio_tag_to_spans(target)
            entities = []
            entity_tags = []
            entity_spans = []
            for tag, (start, end) in spans:
                entities.append(raw_words[start:end])
                entity_tags.append(tag.lower())
                entity_spans.append([start, end])

            ds.append(Instance(raw_words=raw_words, entities=entities, entity_tags=entity_tags,
                               entity_spans=entity_spans))
            if self.demo and len(ds) > 30:
                break
        if len(ds) == 0:
            raise RuntimeError("No data found {}.".format(path))
        return ds


class OntoNotesNERLoader(ConllLoader):
    r"""
    用以读取OntoNotes的NER数据，同时也是Conll2012的NER任务数据。将OntoNote数据处理为conll格式的过程可以参考
    https://github.com/yhcc/OntoNotes-5.0-NER。OntoNoteNERLoader将取第4列和第11列的内容。

    读取的数据格式为：

    Example::

        bc/msnbc/00/msnbc_0000   0   0          Hi   UH   (TOP(FRAG(INTJ*)  -   -   -    Dan_Abrams  *   -
        bc/msnbc/00/msnbc_0000   0   1    everyone   NN              (NP*)  -   -   -    Dan_Abrams  *   -
        ...

    返回的DataSet的内容为

        entities: List[List[str]], 每个元素是entity，非连续的拼到一起了
        entity_tags: 与上面一样长，是每个tag的分数
        raw_words: List[str]词语
        entity_spans： List[List[int]]记录的是上面entity的start和end，这里的长度一定是偶数，是start,end的pair


    """

    def __init__(self, demo=False):
        super().__init__(headers=['raw_words', 'target'], indexes=[3, 10])
        self.demo = demo

    def _load(self, path: str):
        dataset = super()._load(path)

        def convert_to_bio(tags):
            bio_tags = []
            flag = None
            for tag in tags:
                label = tag.strip("()*")
                if '(' in tag:
                    bio_label = 'B-' + label
                    flag = label
                elif flag:
                    bio_label = 'I-' + flag
                else:
                    bio_label = 'O'
                if ')' in tag:
                    flag = None
                bio_tags.append(bio_label)
            return bio_tags

        def convert_word(words):
            converted_words = []
            for word in words:
                word = word.replace('/.', '.')  # 有些结尾的.是/.形式的
                if not word.startswith('-'):
                    converted_words.append(word)
                    continue
                # 以下是由于这些符号被转义了，再转回来
                tfrs = {'-LRB-': '(',
                        '-RRB-': ')',
                        '-LSB-': '[',
                        '-RSB-': ']',
                        '-LCB-': '{',
                        '-RCB-': '}'
                        }
                if word in tfrs:
                    converted_words.append(tfrs[word])
                else:
                    converted_words.append(word)
            return converted_words

        dataset.apply_field(convert_word, field_name=Const.RAW_WORD, new_field_name=Const.RAW_WORD)
        dataset.apply_field(convert_to_bio, field_name=Const.TARGET, new_field_name=Const.TARGET)

        new_dataset = DataSet()

        for ins in dataset:
            raw_words = ins['raw_words']
            target = iob2(ins['target'])
            spans = _bio_tag_to_spans(target)
            entities = []
            entity_tags = []
            entity_spans = []
            for tag, (start, end) in spans:
                entities.append(raw_words[start:end])
                entity_tags.append(tag.lower())
                entity_spans.append([start, end])

            new_dataset.append(Instance(raw_words=raw_words, entities=entities, entity_tags=entity_tags,
                                        entity_spans=entity_spans))

            if len(new_dataset)>30 and self.demo:
                break
        return new_dataset


class DiscontinuousNERLoader(Loader):
    def __init__(self, demo=False):
        super(DiscontinuousNERLoader, self).__init__()
        self.demo = demo

    def _load(self, path):
        """
        entities: List[List[str]], 每个元素是entity，非连续的拼到一起了
        entity_tags: 与上面一样长，是每个tag的分数
        raw_words: List[str]词语
        entity_spans： List[List[int]]记录的是上面entity的start和end，这里的长度一定是偶数，是start,end的pair

        :param path:
        :return:
        """
        max_span_len = 1e10
        f = open(path, 'r', encoding='utf-8')
        lines = f.readlines()
        dataset = DataSet()

        for i in range(len(lines)):
            if i % 3 == 0:
                sentence = lines[i]
                ann = lines[i + 1]
                now_ins = Instance()
                sentence = sentence.strip().split(' ')  # 生成的空格
                entities = ann.strip().split('|')
                type_list = []
                entity_index_list = []
                entity_list = []
                all_spans = []
                for entity in entities:
                    if len(entity) == 0:
                        continue
                    # print(entity)
                    span_, type_ = entity.split(' ')
                    span_ = span_.split(',')
                    span__ = []
                    for i in range(len(span_) // 2):
                        span__.append([int(span_[2 * i]), int(span_[2 * i + 1]) + 1])
                    span__.sort(key=lambda x: x[0])
                    if span__[-1][1] - span__[0][0] > max_span_len:
                        continue
                    str_span__ = []
                    for start, end in span__:
                        str_span__.extend(sentence[start:end])
                    assert len(str_span__) > 0 and len(span__) > 0
                    type_list.append(type_.lower())  # 内部是str
                    entity_list.append(str_span__)
                    entity_index_list.append(list(chain(*span__)))  # 内部是数字
                    all_spans.append([type_.lower(), str_span__, list(chain(*span__))])

                all_spans = sorted(all_spans, key=cmp_to_key(cmp))

                new_type_list = [span[0] for span in all_spans]
                new_entity_list = [span[1] for span in all_spans]
                new_entity_index_list = [span[2] for span in all_spans]

                now_ins.add_field('entities', new_entity_list)
                now_ins.add_field('entity_tags', new_type_list)
                now_ins.add_field('raw_words', sentence)  # 以空格隔开的words
                now_ins.add_field('entity_spans', new_entity_index_list)
                dataset.append(now_ins)
                if self.demo and len(dataset) > 30:
                    break
            else:
                continue

        return dataset


class NestedLoader(Loader):
    def __init__(self, demo=False, **kwargs):
        super().__init__()
        self.demo = demo
        self.max_sent_len = 150

    def _load(self, path):
        def cmp(v1, v2):
            v1 = v1[1]
            v2 = v2[1]
            if v1[0] == v2[0]:
                return v1[1] - v2[1]
            return v1[0] - v2[0]

        ds = DataSet()
        invalid_ent = 0
        max_len = 0
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines, total=len(lines), leave=False):
                data = json.loads(line.strip())

                all_entities = data['ners']
                all_sentences = data['sentences']

                assert len(all_entities) == len(all_sentences)

                # TODO 这里，一句话不要超过100个词吧
                new_all_sentences = []
                new_all_entities = []
                for idx, sent in enumerate(all_sentences):
                    ents = all_entities[idx]
                    if len(sent)>self.max_sent_len:
                        has_entity_cross = np.zeros(len(sent))
                        for (start, end, tag) in ents:
                            has_entity_cross[start:end + 1] = 1  # 如果1为1的地方说明是有span穿过的

                        punct_indexes = []
                        for idx, word in enumerate(sent):
                            if word.endswith('.') and has_entity_cross[idx]==0:
                                punct_indexes.append(idx)
                        last_index = 0
                        for idx in punct_indexes:
                            if idx-last_index>self.max_sent_len:
                                new_all_sentences.append(sent[last_index:idx+1])
                                new_ents = [(e[0]-last_index, e[1]-last_index, e[2]) for e in ents if last_index<=e[1]<=idx]  # 是闭区间
                                new_all_entities.append(new_ents)
                                last_index = idx+1
                        if last_index<len(sent):
                            new_all_sentences.append(sent[last_index:])
                            new_ents = [(e[0]-last_index, e[1]-last_index, e[2]) for e in ents if last_index <= e[1]]  # 是闭区间
                            new_all_entities.append(new_ents)
                    else:
                        new_all_sentences.append(sent)
                        new_all_entities.append(ents)
                if sum(map(len, all_entities))!=sum(map(len, new_all_entities)):
                    print("Mismatch number sentences")
                if sum(map(len, all_sentences))!=sum(map(len, new_all_sentences)):
                    print("Mismatch number entities")

                all_entities = new_all_entities
                all_sentences = new_all_sentences

                for i in range(len(all_entities)):
                    all_spans = []
                    raw_words = all_sentences[i]
                    max_len = max(len(raw_words), max_len)
                    ents = all_entities[i]
                    for start, end, tag in ents:
                        # assert start<=end, (start, end, i)
                        if start>end:
                            invalid_ent += 1
                            continue
                        all_spans.append((tag, (start, end+1)))
                        assert end<len(raw_words), (end, len(raw_words))

                    all_spans = sorted(all_spans, key=cmp_to_key(cmp))

                    entities = []
                    entity_tags = []
                    entity_spans = []
                    for tag, (start, end) in all_spans:
                        entities.append(raw_words[start:end])
                        entity_tags.append(tag.lower())
                        entity_spans.append([start, end])

                    prev_contxt = []
                    after_contxt = []

                    if i>0:
                        prev_contxt = all_sentences[:i]
                    if i<len(all_sentences)-1:
                        after_contxt = all_sentences[i+1:]

                    assert len(after_contxt)+len(prev_contxt)==len(all_sentences)-1

                    ds.append(Instance(raw_words=raw_words, entities=entities, entity_tags=entity_tags,
                                       entity_spans=entity_spans,
                                       prev_contxt=prev_contxt, after_contxt=after_contxt))
                if self.demo and len(ds) > 30:
                    break
        if len(ds) == 0:
            raise RuntimeError("No data found {}.".format(path))
        print(f"for `{path}`, {invalid_ent} invalid entities. max sentence has {max_len} tokens")
        return ds



def cmp(v1, v2):
    v1 = v1[-1]
    v2 = v2[-1]
    if v1[0] == v2[0]:
        return v1[-1] - v2[-1]
    return v1[0] - v2[0]


if __name__ == '__main__':
    data_bundle = Conll2003NERLoader(demo=False).load('data/conll2003')
    BartNERPipe(target_type='word', dataset_name='conll2003').process(data_bundle)
