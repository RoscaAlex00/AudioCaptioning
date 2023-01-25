from pathlib import Path
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from torch.utils.data import DataLoader

from typing import List, Tuple
from pathlib import Path

import torch.utils.data
import numpy as np
from yamnet_input import class_names_from_csv
from metadata import load_pickle_file

from transformers import BartTokenizerFast
import string


class AACDatasetBART(torch.utils.data.Dataset):
    def __init__(self, settings, data_dir, split, tokenizer):
        super(AACDatasetBART, self).__init__()
        the_dir = data_dir.joinpath(split)

        self.examples = sorted(the_dir.iterdir())

        self.max_audio_len = settings['data']['max_audio_len']
        self.max_caption_tok_len = settings['data']['max_caption_tok_len']
        self.input_name = settings['data']['input_field_name']
        self.output_name = settings['data']['output_field_name']
        self.cond_tok_field_name = settings['data']['cond_tok_field_name']
        self.cond_tok_class_sel = settings['data']['cond_tok_class_sel']
        self.cond_tok_time_sel = settings['data']['cond_tok_time_sel']
        self.cond_tok_separator = settings['data']['cond_tok_separator']
        self.metadata = settings['lm']['config']['metadata']

        if self.cond_tok_field_name is not None and 'logits' in self.cond_tok_field_name and not self.metadata:
            self.class_map = class_names_from_csv()

        print(data_dir)
        if self.metadata:
            self.metadata_keywords = load_pickle_file(data_dir.joinpath(f'{split}_keywords_dict_metadata.p'))

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        ex = self.examples[item]
        ex = np.load(str(ex), allow_pickle=True)

        # ----- Labels/Decoder inputs -----
        ou_e = ex[self.output_name].item()

        if ou_e is not None:
            ou_e = ou_e.translate(str.maketrans('', '', string.punctuation))
            ou_e = ou_e.lower()

            tok_e = self.tokenizer(ou_e, max_length=self.max_caption_tok_len, return_tensors='pt', padding='max_length')
            if tok_e['input_ids'].size(1) > self.max_caption_tok_len:
                print('Found caption longer than max_caption_tok_len parameter ({} tokens).'.format(
                    tok_e['input_ids'].size(1)))
                tok_e['input_ids'] = tok_e['input_ids'][:, :self.max_caption_tok_len]
                tok_e['attention_mask'] = tok_e['attention_mask'][:, :self.max_caption_tok_len]
        else:
            tok_e = {'input_ids': None, 'attention_mask': None}

        # ----- Audio conditioning -----
        in_e = ex[self.input_name].item()

        in_e = torch.Tensor(in_e).float().unsqueeze(0)

        in_e = in_e.squeeze()
        if len(list(in_e.size())) == 1:  # Single embedding in sequence
            in_e = in_e.unsqueeze(0)

        # ----- Conditioning inputs -----
        cond_text = None
        if self.cond_tok_field_name is not None and not self.metadata:
            if 'logits' in self.cond_tok_field_name:
                cond_tok_logits = torch.Tensor(ex[self.cond_tok_field_name].item()).float()
                # ----- Tag sampling -----
                if 'top' in self.cond_tok_time_sel:  # Eg: 'top5'
                    if cond_tok_logits.size(0) == 1 and in_e.size(0) != 1:
                        cond_tok_logits = cond_tok_logits.repeat(in_e.size(0), 1)
                    cond_tok_logits = torch.mean(cond_tok_logits, dim=0)  # Average along time
                    cond_tok_class = torch.argsort(cond_tok_logits, descending=True)[:int(self.cond_tok_time_sel[3:])]
                else:  # Num tags = num logits
                    cond_tok_class = torch.multinomial(cond_tok_logits, 1)
                    if self.cond_tok_time_sel == 'unroll' and in_e.size(0) == cond_tok_class.size(0) - 1:  # Some 9.5s files, vggish cuts whereas yamnet pads
                        cond_tok_class = cond_tok_class[:in_e.size(0)]
                len_cond_tok = cond_tok_class.size(0)
                # ----- Reformat text inputs -----
                cond_text = self.cond_tok_separator.join([self.class_map[int(ic)] for ic in cond_tok_class]) + '.'
                # print(cond_text)
            cond_tokens = self.tokenizer(cond_text, max_length=64, return_tensors='pt',
                                         padding='max_length')  # Reduce to 128 on AudioCaps for speedup
            att_mask = cond_tokens['attention_mask'].squeeze()
            cond_tokens = cond_tokens['input_ids'].squeeze().long()
        else:
            cond_tokens = None
            att_mask = None

        if self.metadata:
            cond_text = self.cond_tok_separator.join(self.metadata_keywords[ex['file_name'].item()]) + '.'
            # print(cond_text)
            cond_tokens = self.tokenizer(cond_text, max_length=64, return_tensors='pt',
                                         padding='max_length')  # Reduce to 128 on AudioCaps for speedup
            att_mask = cond_tokens['attention_mask'].squeeze()
            cond_tokens = cond_tokens['input_ids'].squeeze().long()

        # ----- Reformat audio inputs -----
        if self.cond_tok_field_name is None:  # No token cond
            audio_att_mask = torch.zeros((self.max_audio_len,)).long()

            len_in_e = in_e.size(0)

            audio_att_mask[:len_in_e] = 1
            if in_e.size(0) > self.max_audio_len:
                in_e = in_e[:self.max_audio_len, :]
            elif in_e.size(0) < self.max_audio_len:
                in_e = torch.cat([in_e, torch.zeros(self.max_audio_len - in_e.size(0),
                                                    in_e.size(1)).float()])  # BART encoder max_length = 1024 ?
        else:
            audio_att_mask = None
            if 'top' in self.cond_tok_time_sel:  # Eg: 'top5'
                if in_e.size(0) != 1:
                    in_e = in_e.mean(dim=0, keepdim=True)
                in_e = in_e[0, :].repeat(cond_tokens.size(0), 1)
            else:  # Num tags = num logits
                if in_e.size(0) != len_cond_tok and in_e.size(0) == 1:  # Eg.: yamnet logits with panns embeddings
                    in_e = in_e.repeat(len_cond_tok, 1)
                if in_e.size(0) != len_cond_tok and int(
                        np.ceil(len_cond_tok / 10.) / in_e.size(0)) == 1:  # Panns for Clotho
                    in_e = in_e.repeat_interleave(10, dim=0)
                    in_e = in_e[:len_cond_tok, :]
                    if in_e.size(0) < len_cond_tok:  # e.g. 30 < 31 in some cases
                        in_e = torch.cat((in_e, in_e[-1, :].repeat(len_cond_tok - in_e.size(0), 1)), dim=0)
                if in_e.size(0) != len_cond_tok and in_e.size(0) == 2:
                    in_e = in_e[0, :].repeat(len_cond_tok, 1)
                assert in_e.size(
                    0) == len_cond_tok, 'Audio embeddings ({}) and tags ({}) dimensions do not match for file {}.'.format(
                    in_e.size(0), len_cond_tok, ex['file_name'].item())
                sep_token = \
                    self.tokenizer.encode('and' + self.cond_tok_separator, return_tensors='pt',
                                          add_special_tokens=False)[
                        0, 1]
                audio_features = torch.zeros((att_mask.size(0), in_e.size(1)))
                i_frame = 0
                for i_tok in range(att_mask.size(0)):
                    if cond_tokens[i_tok] == 1 or cond_tokens[i_tok] == 0 or cond_tokens[i_tok] == 2:
                        pass  # BOS, EOS, PAD
                    elif cond_tokens[i_tok] == sep_token:  # separator
                        i_frame += 1
                    else:
                        audio_features[i_tok, :] = in_e[i_frame, :]
                in_e = audio_features

        return {'audio_features': in_e,
                'attention_mask': audio_att_mask,
                'decoder_attention_mask': tok_e['attention_mask'].squeeze() if tok_e[
                                                                                   'attention_mask'] is not None else None,
                'file_name': ex['file_name'].item(),
                'labels': tok_e['input_ids'].squeeze().long() if tok_e['input_ids'] is not None else None,
                'cond_tokens': cond_tokens,
                'cond_text': cond_text
                }


# Modification of the transformers default_data_collator function to allow string and list inputs
InputDataClass = NewType("InputDataClass", Any)


def default_data_collator(features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
    """
    Very simple data collator that simply collates batches of dict-like objects and performs special handling for
    potential keys named:
        - ``label``: handles a single value (int or float) per object
        - ``label_ids``: handles a list of values per object
    Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs
    to the model. See glue and ner for example of how it's useful.
    """

    # In this function we'll make the assumption that all `features` in the batch
    # have the same attributes.
    # So we will look at the first element as a proxy for what attributes exist
    # on the whole batch.

    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, List) and v != [] and isinstance(v[0], str):
                batch[k] = [f[k] for f in features]
            else:
                batch[k] = torch.tensor([f[k] for f in features])
        elif k not in ("label", "label_ids") and v is not None:  # str
            batch[k] = [f[k] for f in features]

    return batch


def get_dataset(split, settings, tokenizer):
    data_dir = Path(settings['data']['root_dir'], settings['data']['features_dir'])
    if split == 'training' and settings['workflow']['validate']:
        return AACDatasetBART(settings, data_dir, 'development', tokenizer), \
               AACDatasetBART(settings, data_dir, 'validation', tokenizer)
    else:
        return AACDatasetBART(settings, data_dir, split, tokenizer), None
