from pathlib import Path
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from torch.utils.data import DataLoader
import torch

from typing import List, Tuple
from pathlib import Path

import torch.utils.data
import numpy as np
import pandas as pd

from transformers import BartTokenizerFast
import string

class AACDatasetBART(torch.utils.data.Dataset):
    def __init__(self, settings, data_dir : Path, split, tokenizer):
        super(AACDatasetBART, self).__init__()
        the_dir = data_dir.joinpath(split)
        
        self.examples = sorted(the_dir.iterdir())
        
        self.max_audio_len = settings['data']['max_audio_len']
        self.max_caption_tok_len = settings['data']['max_caption_tok_len']
        self.input_name = settings['data']['input_field_name']
        self.output_name = settings['data']['output_field_name']
        self.audio_input = settings["model_inputs"]["audio_features"]
        self.keyword_input = settings["model_inputs"]["keywords"]
        self.keyword_dir = settings["data"]["keyword_dir"]
        self.max_keywords = settings["model_inputs"]["max_keywords"]

        if self.keyword_dir is None:
            # Set keyword input to false, as there is no directory given to get the keywords from
            self.keyword_input = False

        if self.keyword_input:
            metadata_dir = Path(settings['data']['root_dir'], settings['data']['keyword_dir'])
            self.keyword_csv = pd.read_csv(metadata_dir.joinpath("clotho_metadata_" + split + ".csv"), encoding='unicode_escape')
            if "keywords" not in self.keyword_csv.columns:
                print(f"No keywords found for {split} split. keyword_input will be put to False")
                self.keyword_input = False
            else:
                self.keyword_csv = self.keyword_csv[["file_name", "keywords"]] # only need these columns
                
        if not (self.audio_input or self.keyword_input):
            raise Exception("Need at least one model input to be true")

        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, item):
        ex = self.examples[item]
        fname = Path(ex).name
        ex = np.load(str(ex), allow_pickle=True)
        
        # ----- Labels/Decoder inputs -----
        ou_e = ex[self.output_name].item()
        
        if ou_e is not None:
            ou_e = ou_e.translate(str.maketrans('', '', string.punctuation))
            ou_e = ou_e.lower()
            
            tok_e = self.tokenizer(ou_e, max_length=self.max_caption_tok_len, return_tensors='pt', padding='max_length')
            if tok_e['input_ids'].size(1) > self.max_caption_tok_len:
                print('Found caption longer than max_caption_tok_len parameter ({} tokens).'.format(tok_e['input_ids'].size(1)))
                tok_e['input_ids'] = tok_e['input_ids'][:,:self.max_caption_tok_len]
                tok_e['attention_mask'] = tok_e['attention_mask'][:,:self.max_caption_tok_len]
        else:
            tok_e = {'input_ids': None, 'attention_mask': None}

        # ----- Keywords -----
        if self.keyword_input:
            # Extracting corresponding keywords
            restoredfname = fname[7:-6] # remove clotho_ prefix and _x.npy suffix
            kwords = self.keyword_csv[self.keyword_csv["file_name"] == restoredfname]["keywords"]
            kwords = kwords.iloc[0]
            kwords = " ".join(kwords.split(";"))
            tok_kws = self.tokenizer(kwords, max_length=self.max_keywords, return_tensors="pt", padding="max_length")
            kws_mask = tok_kws["attention_mask"]
            tok_kws = tok_kws["input_ids"]
            if tok_kws.size(1) > self.max_keywords:
                tok_kws = tok_kws[:, :self.max_keywords]
                kws_mask = kws_mask[:, :self.max_keywords]

            # Adjusting the decoder attention mask
            decoder_attention_mask = torch.cat((kws_mask.squeeze(), torch.ones((1, ), dtype=torch.long), tok_e["attention_mask"].squeeze()))
            tok_e["attention_mask"] = decoder_attention_mask.unsqueeze(dim=0)

            # Adjusting the labels (tokens with -100 are ignored, so no loss computation on those)
            labels = torch.cat((torch.full((1, self.max_keywords + 1), -100, dtype=torch.long), tok_e["input_ids"]), dim=-1)
            tok_e["input_ids"] = labels
        

        if self.audio_input:
            # ----- Audio conditioning -----
            in_e = ex[self.input_name].item()
            
            in_e = torch.Tensor(in_e).float().unsqueeze(0)
            
            in_e = in_e.squeeze()
            if len(list(in_e.size())) == 1: # Single embedding in sequence
                in_e = in_e.unsqueeze(0)
        
            # ----- Reformat audio inputs -----
            audio_att_mask = torch.zeros((self.max_audio_len,)).long()
            
            audio_att_mask[:in_e.size(0)] = 1
            if in_e.size(0) > self.max_audio_len:
                in_e = in_e[:self.max_audio_len, :]
            elif in_e.size(0) < self.max_audio_len:
                in_e = torch.cat([in_e, torch.zeros(self.max_audio_len - in_e.size(0), in_e.size(1)).float()])
        else:
            in_e = None
            audio_att_mask = None
        
        return {'audio_features': in_e,
                'attention_mask': audio_att_mask,
                'decoder_attention_mask': tok_e['attention_mask'].squeeze() if tok_e['attention_mask'] is not None else None,
                'file_name': ex['file_name'].item(),
                'labels': tok_e['input_ids'].squeeze().long() if tok_e['input_ids'] is not None else None,
                'decoder_input_ids': tok_kws.squeeze() if self.keyword_input else None}



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
        elif k not in ("label", "label_ids") and v is not None: # str
            batch[k] = [f[k] for f in features]
    
    return batch

def get_dataset(split, settings, tokenizer):
    data_dir = Path(settings['data']['root_dir'], settings['data']['features_dir'])
    if split == 'training' and settings['workflow']['validate']:
        return AACDatasetBART(settings, data_dir, 'development', tokenizer), \
               AACDatasetBART(settings, data_dir, 'validation', tokenizer)
    else:
        return AACDatasetBART(settings, data_dir, split, tokenizer), None
        
