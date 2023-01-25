# AudioCaptioning
## Transformer-based audio captioning
This code can be used to train an AAC model. You can specify whether the model should get audio, audio and keywords, or only keywords as input.

To set up the repository, please follow the instructions at https://github.com/felixgontier/dcase-2022-baseline. This repository was used as baseline.

In order to run experiments, run
```
python main.py --exp <experiment_name>
```

The configurations of the various models are located in the `exp_settings` folder. 
