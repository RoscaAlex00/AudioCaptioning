from models import BARTAAC
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BartConfig, BartForConditionalGeneration
from transformers.models.bart.modeling_bart import BartEncoder
from transformers.modeling_outputs import BaseModelOutput

from torch.nn import Linear, LayerNorm
from transformers.models.bart.modeling_bart import BartAttention