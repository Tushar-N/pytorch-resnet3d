# 3D ConvNets in Pytorch

Do you want 72% top-1 accuracy on a large video dataset? Are you tired of Kinetics videos disappearing from YouTube every day? Do you have recurring nightmares about Caffe2? Then this is the repo for you!

This is a PyTorch implementation of the Caffe2 I3D ResNet baseline from the [video-nonlocal-net](https://github.com/facebookresearch/video-nonlocal-net) repo. The weights are directly ported from the caffe2 model (See [checkpoints](https://github.com/facebookresearch/video-nonlocal-net#main-results)). This should be a good starting point to extract features, finetune on another dataset etc. without the hassle of dealing with Caffe2, and with all the benefits of a very carefully trained Kinetics model. 

It's only a matter of time before FAIR releases a good PyTorch version of their non-local-net codebase, but until then, at least you have this ¯\\\_(ツ)\_/¯

**Amazing features**: 
&#8291;- Only a single model (ResNet50-I3D). Parameters hardcoded with love. 
&#8291;- Only the evaluation script for Kinetics (training from scratch or ftuning has not been tested yet.)
&#8291;- No non-local versions yet. 


## Kinetics Evaluation

The code has been tested with Python 3.7 + PyTorch 1.0.

**Pretrained Weights**
Download pretrained weights for `run_i3d_baseline_400k_32f` using:
```bash
bash pretrained/download_weights.sh
```
Convert these weights from caffe2 to pytorch. This is just a simple renaming of the blobs to match the pytorch model.
```bash
python -m utils.convert_weights pretrained/i3d_baseline_32x2_IN_pretrain_400k.pkl pretrained/i3d_r50_kinetics.pth
```

The model can be created and weights loaded using
```python
from models import resnet
net = resnet.i3_res50()
```

**Data** 
Download videos using the [official crawler](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics) and extract frames. [This repo](https://github.com/kenshohara/3D-ResNets-PyTorch/#kinetics) has a script to do this.

create softlinks for frames and annotations
```bash
mkdir -p data/kinetics/frames/ data/kinetics/annotations/
ln -s /path/to/kinetics/frames data/kinetics/frames/
ln -s /path/to/kinetics/annotation_csvs data/kinetics/annotations/
```

**Evaluate**
Run the evaluation script to generate scores on the validation set. 
```bash
# Evaluation using 3 random spatial crops per frame + 10 uniformly sampled clips per video
python eval.py --batch_size 8 --mode video
>> (test) A: 0.722 | clf: 1.158 | total_loss: 1.158

# Evaluation using a single, center crop and a single, centered clip of 32 frames
python eval.py --batch_size 8 --mode clip
>> (test) A: 0.647 | clf: 1.551 | total_loss: 1.551

# Use --parallel for multiple GPUs
python eval.py --batch_size 64 --mode clip --parallel
```

You should get around 72.2% top-1 accuracy for the video (3 random spatial crops per frame + 10 uniformly sampled clips per video) and around 64.7% top-1 accuracy for the clip (single, center crop and a single, centered clip). Note that these numbers are on whatever is left of the Kinetics val set these days (~18434 videos).