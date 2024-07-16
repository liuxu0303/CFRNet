## High-rate monocular depth estimation via cross frame-rate collaboration of frames and events

![avatar](Pictures/framework.jpg)

The pipeline of the proposed high-rate monocular depth estimator using a *cross frame-rate frame-event joint learning network (**CFRNet**)*. The continuous event stream is first split into high-rate event temporal bins and converted into event representations. For the current timestamp $t_i$, we use the modal-specific shared encoder adopting a lightweight CNN-Transformer hybrid backbone (*i.e.,* lite-mono) to extract local-global features from the event representation $E_i$ and the most recent frame $F_j$, respectively. Then, the proposed CFMF utilizes implicit spatial alignment and dynamic attention-based fusion strategies to generate a complementary joint representation. Meanwhile, a novel  recurrent network TCM effectively models long-range temporal dependencies between the joint representations. Finally, a normal CNN-based decoder predicts high-rate and fine-grained depth maps.

#### Dependencies

PyTorch >= 1.0
NumPy
OpenCV
Matplotlib

### Setup
This code has been tested with Python 3.7.10, Torch 1.9.0, CUDA 10.2 on Ubuntu 16.04.

- Setup python environment
```
conda create -n EReFormer python=3.7.10
source activate EReFormer 
pip install -r requirements.txt
conda install -c pytorch pytorch=1.9.0 torchvision cudatoolkit=10.2
conda install -c conda-forge opencv
conda install -c conda-forge matplotlib
```

### Public Datasets

We use a widely adopted event-based vision dataset (*i.e.*, DSEC) to validate our novel framework CFRNet, which contains challenging illumination conditions such as night, sunrise, and sunset. Since DSEC is not specifically designed for monocular depth estimation, we follow the guidance the DSEC benchmark provides and convert disparity ground truth to depth. We manually split the training data into train and test subsets, dividing the daytime and nighttime sequences into two subsets, respectively. For the training of CFRNet, two different frame rates of frames $\boldsymbol{F} = \left\lbrace F_{1}, ..., F_{T}\right\rbrace$ and event temporal bins $\boldsymbol{S} = \left\lbrace S_{1}, ..., S_{K}\right\rbrace$ are required. Therefore, we keep the frames rate (\textit{i.e.,} 2.5 Hz) at $1/4$ of the event temporal bins frame rate (\textit{i.e.,} depth ground truth frame rate, 10 Hz). The training set consists of 35 sequences with more than 5.4k frames, and 21.9k depth ground-truth and corresponding recorded events. The testing set consists of 6 sequences with more than 1k frames, 4.2k depth ground-truth and corresponding recorded events. During inference, we achieve high-rate monocular depth estimation by increasing the input rate of frames and event temporal bins, especially by increasing the rate of the latter. In addition, we also discuss the effectiveness of CFRNet in handling two modalities with the same frame rate on the MVSEC dataset. For MVSEC, we follow the experimental settings of previous works, training the monocular depth estimation methods on the outdoor day2 sequence and evaluating them on the outdoor day1 and outdoor night1 sequences.

High-rate monocular depth estimation at 100Hz, where the frame modality frame rate is 2.5Hz and the event modality is 100Hz:

![High-rate monocular depth estimation at 100Hz](https://github.com/liuxu0303/CFRNet/blob/main/High_rate_depth_100Hz.gif)
