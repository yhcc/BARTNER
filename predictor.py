
import torch
from fastNLP import cache_results
from data.pipe import BartNERPipe
from fastNLP import SequentialSampler, SortedSampler

from fastNLP import DataSetIter
from fastNLP.core.utils import _move_dict_value_to_device
from tqdm import tqdm


dataset_name = 'conll2003'
model_path = 'save_models/best_SequenceGeneratorModel_f_2021-06-09-01-47-26-903275'  # you can set args.save_model=1 in train.py
bart_name = 'facebook/bart-large'
target_type = 'word'
device = 'cuda:0'

cache_fn = f"caches/data_{bart_name}_{dataset_name}_{target_type}.pt"

@cache_results(cache_fn, _refresh=False)
def get_data():
    pipe = BartNERPipe(tokenizer=bart_name, dataset_name=dataset_name, target_type=target_type)
    if dataset_name == 'conll2003':
        paths = {'test': "../data/conll2003/test.txt",
                 'train': "../data/conll2003/train.txt",
                 'dev': "../data/conll2003/dev.txt"}
        data_bundle = pipe.process_from_file(paths, demo=False)
    elif dataset_name == 'en-ontonotes':
        paths = '../data/en-ontonotes/english'
        data_bundle = pipe.process_from_file(paths)
    else:
        data_bundle = pipe.process_from_file(f'../data/{dataset_name}', demo=False)
    return data_bundle, pipe.tokenizer, pipe.mapping2id

data_bundle, tokenizer, mapping2id = get_data()

model = torch.load(model_path)

device = torch.device(device)
model.to(device)
model.eval()

eos_token_id = 0
word_start_index = len(mapping2id)+2
not_bpe_start = 0
if dataset_name == 'conll2003':  # if you use other dataset, please change this mapping
    mapping = {
        '<<location>>': 'LOC',
        '<<person>>': 'PER',
        '<<organization>>': 'ORG',
        '<<others>>': 'MISC'
    }

id2label = {k:mapping[v] for k,v in enumerate(mapping2id.keys())}
for name in ['test']:
    ds = data_bundle.get_dataset(name)
    ds.set_ignore_type('raw_words', 'raw_target')
    ds.set_target('raw_words', 'raw_target')
    with open(f'preds/{name}.conll', 'w', encoding='utf-8') as f:
        data_iterator = DataSetIter(ds, batch_size=32, sampler=SequentialSampler())
        for batch_x, batch_y in tqdm(data_iterator, total=len(data_iterator)):
            _move_dict_value_to_device(batch_x, batch_y, device=device)
            src_tokens = batch_x['src_tokens']
            first = batch_x['first']
            src_seq_len = batch_x['src_seq_len']
            raw_words = batch_y['raw_words']
            raw_targets = batch_y['raw_target']
            pred_y = model.predict(src_tokens=src_tokens, src_seq_len=src_seq_len, first=first)
            pred = pred_y['pred']
            pred_eos_index = pred.flip(dims=[1]).eq(eos_token_id).cumsum(dim=1).long()
            pred = pred[:, 1:]  # 去掉</s>
            pred_seq_len = pred_eos_index.flip(dims=[1]).eq(pred_eos_index[:, -1:]).sum(dim=1)  # bsz
            pred_seq_len = (pred_seq_len - 2).tolist()
            for i, ps in enumerate(pred.tolist()):
                em = 0
                ps = ps[:pred_seq_len[i]]
                pairs = []
                cur_pair = []
                if len(ps):
                    for j in ps:
                        if j < word_start_index:
                            if target_type == 'span':
                                if len(cur_pair) > 0 and len(cur_pair) % 2 == 0:
                                    if all([cur_pair[i] <= cur_pair[i + 1] for i in range(len(cur_pair) - 1)]):
                                        pairs.append(tuple(cur_pair + [j]))
                            else:
                                if len(cur_pair) > 0:
                                    if all([cur_pair[i] < cur_pair[i + 1] for i in range(len(cur_pair) - 1)]):
                                        pairs.append(tuple(cur_pair + [j]))
                            cur_pair = []
                        else:
                            cur_pair.append(j)
                raw_words_i = raw_words[i]
                src_tokens_i = src_tokens[i, :src_seq_len[i]].tolist()
                src_tokens_i = tokenizer.convert_ids_to_tokens(src_tokens_i)
                cum_lens = [1]
                start_idx = 1
                for token in raw_words_i:
                    start_idx += len(tokenizer.tokenize(token, add_prefix_space=True))
                    cum_lens.append(start_idx)
                cum_lens.append(start_idx+1)
                pred_y = ['O' for _ in range(len(raw_words_i))]
                target_y = raw_targets[i]
                for pair in pairs:
                    label = pair[-1]
                    try:
                        idxes = [cum_lens.index(p-len(mapping2id)-2) for p in pair[:-1]]
                        start_idx = idxes[0]
                        end_idx = idxes[-1]
                        pred_y[start_idx] = f'B-{id2label[label-2]}'
                        for _ in range(start_idx+1, end_idx+1):
                            pred_y[_] = f'I-{id2label[label-2]}'
                    except Exception as e:
                        not_bpe_start += 1
                assert len(pred_y)==len(raw_words_i)==len(target_y)
                for raw_word, t, p in zip(raw_words_i, target_y, pred_y):
                    f.write(f'{raw_word} {t} {p}\n')
                f.write('\n')

print(f"In total, has {not_bpe_start} predictions on the non-word start.")

# the output file will be similar to the following(empty line separate two sentences)
# SOCCER O O
# - O O
# JAPAN B-LOC B-LOC
# GET O O
# LUCKY O O
# WIN O O
# , O O
# CHINA B-PER B-LOC