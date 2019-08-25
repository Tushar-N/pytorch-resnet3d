# 3D ConvNets in Pytorch

Do you want >72% top-1 accuracy on a large video dataset? Are you tired of Kinetics videos disappearing from YouTube every day? Do you have recurring nightmares about Caffe2? Then this is the repo for you!

This is a PyTorch implementation of the Caffe2 I3D ResNet Nonlocal model from the [video-nonlocal-net](https://github.com/facebookresearch/video-nonlocal-net) repo. The weights are directly ported from the caffe2 model (See [checkpoints](https://github.com/facebookresearch/video-nonlocal-net#main-results)). This should be a good starting point to extract features, finetune on another dataset etc. without the hassle of dealing with Caffe2, and with all the benefits of a very carefully trained Kinetics model. 

It's only a matter of time before FAIR releases a good PyTorch version of their nonlocal-net codebase, but until then, at least you have this ¯\\\_(ツ)\_/¯

**Amazing features**:  
&#8291;- Only a single model (ResNet50-I3D). Parameters hardcoded with love.  
&#8291;- Only the evaluation script for Kinetics (training from scratch or ftuning has not been tested yet.)  
&#8291;- ~~No nonlocal versions yet.~~ One exciting NL version to choose from.


## Kinetics Evaluation

The code has been tested with Python 3.7 + PyTorch 1.1.

**Pretrained Weights**  
Download pretrained weights for I3D and I3D-NL models from the nonlocal repo
```bash
wget https://dl.fbaipublicfiles.com/video-nonlocal/i3d_baseline_32x2_IN_pretrain_400k.pkl -P pretrained/
wget https://dl.fbaipublicfiles.com/video-nonlocal/i3d_nonlocal_32x2_IN_pretrain_400k.pkl -P pretrained/
```
Convert these weights from caffe2 to pytorch. This is just a simple renaming of the blobs to match the pytorch model.
```bash
python -m utils.convert_weights pretrained/i3d_baseline_32x2_IN_pretrain_400k.pkl pretrained/i3d_r50_kinetics.pth
python -m utils.convert_weights pretrained/i3d_nonlocal_32x2_IN_pretrain_400k.pkl pretrained/i3d_r50_nl_kinetics.pth
```

The model can be created and weights loaded using
```python
from models import resnet
net = resnet.i3_res50() # vanilla I3D ResNet50
net = resnet.i3_res50_nl() # Nonlocal version
```

**Data**   
Download videos using the [official crawler](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics) and extract frames. [This repo](https://github.com/kenshohara/3D-ResNets-PyTorch/#kinetics) has a script to do this. Then create softlinks for frames and annotations:
```bash
mkdir -p data/kinetics/frames/ data/kinetics/annotations/
ln -s /path/to/kinetics/frames data/kinetics/frames/
ln -s /path/to/kinetics/annotation_csvs data/kinetics/annotations/
```

**Evaluate**  
Run the evaluation script to generate scores on the validation set. 
```bash
# Evaluate using 3 random spatial crops per frame + 10 uniformly sampled clips per video
# Model = I3D ResNet50 Nonlocal
python eval.py --batch_size 8 --mode video --model r50_nl

# Evaluate using a single, center crop and a single, centered clip of 32 frames 
# Model = I3D ResNet50
python eval.py --batch_size 8 --mode clip --model r50

# Use --parallel for multiple GPUs
python eval.py --batch_size 16 --mode clip --model r50_nl --parallel

```

| Model        | clip (top1/top5)  | video (top1/top5) |
|--------------|-------------------|-------------------|
| I3D Res50    | 0.647 / 0.853     | 0.721 / 0.902     |    
| I3D Res50 NL | 0.664 / 0.864     | 0.737 / 0.912     |


You should get around **72.1%** top-1 accuracy for the video using I3D Res50, and around **73.7%** using the non-local version. Note that these numbers are on whatever is left of the Kinetics val set these days (~18434 videos).
