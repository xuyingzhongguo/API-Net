# Learning Pairwise Interaction for Generalizable DeepFake Detection
Ying Xu, Kiran Raja, Luisa Verdoliva, Marius Pedersen
# Introduction:
We propose a new approach, Multi-Channel Xception Attention Pairwise Interaction (MCX-API), that exploits the power of pairwise learning and complementary information from different color space representations in a fine-grained manner. We first validate our idea on a publicly available dataset in a intra-class setting (closed set) with four different Deepfake schemes. Further, we report all the results using balanced-open-set-classification (BOSC) accuracy in an inter-class setting (open-set) using three public datasets. Our experiments indicate that our proposed method can generalize better than the state-of-the-art Deepfakes detectors. We obtain 98.48% BOSC accuracy on the FF++ dataset and 90.87% BOSC accuracy on the CelebDF dataset suggesting a promising direction for generalization of DeepFake detection. We further utilize t-SNE and attention maps to interpret and visualize the decision-making process of our proposed network.
# Framework:
![Framework](/figures/mcx-api.jpeg)

# How to use:
If you want to test, please refer to [test.slurm](test.slurm) for examples.

# Download model
Here is the [link](https://drive.google.com/drive/folders/1jMdXLp3LhG06YQQicRu00aducCa2hcOT?usp=sharing) for MCX-API model for RGB. 

# Citing:
Please kindly cite the following paper, if you find this code helpful in your work.
```
@inproceedings{zhuang2020learning,
  title={Learning Attentive Pairwise Interaction for Fine-Grained Classification.},
  author={Zhuang, Peiqin and Wang, Yali and Qiao, Yu},
  booktitle={AAAI},
  pages={13130--13137},
  year={2020}
}
```
# Contact:
Please feel free to contact ying.xu@ntnu.no, if you have any questions.


