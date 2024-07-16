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

We use a widely adopted event-based vision dataset (*i.e.*, DSEC) to validate our novel framework CFRNet, which contains challenging illumination conditions such as night, sunrise, and sunset. Since DSEC is not specifically designed for monocular depth estimation, we follow the guidance the DSEC benchmark provides and convert disparity ground truth to depth. We manually split the training data into train and test subsets, dividing the daytime and nighttime sequences into two subsets, respectively. 

DSEC: A Stereo Event Camera Dataset for Driving Scenarios: https://dsec.ifi.uzh.ch/dsec-datasets/download/

- Training Sequences
```
interlaken_00_c, interlaken_00_d, interlaken_00_e, interlaken_00_f, zurich_city_00_a, zurich_city_00_b, zurich_city_01_a, zurich_city_01_b, zurich_city_01_c, zurich_city_01_d, zurich_city_01_e, zurich_city_01_f, zurich_city_02_a, zurich_city_02_b, zurich_city_02_c, zurich_city_03_a, zurich_city_04_a, zurich_city_04_b, zurich_city_04_c, zurich_city_04_d, zurich_city_04_e, zurich_city_04_f, zurich_city_05_a, zurich_city_05_b, zurich_city_07_a, zurich_city_08_a, zurich_city_09_b, zurich_city_09_c, zurich_city_09_d, zurich_city_09_e, zurich_city_10_a, zurich_city_10_b, zurich_city_11_a, zurich_city_11_b, zurich_city_11_c
```

- Testing Sequences
```
interlaken_00_g, thun_00_a, zurich_city_02_d, zurich_city_02_e, zurich_city_06_a, zurich_city_09_a
```

For the training of CFRNet, two different frame rates of frames $\boldsymbol{F} = \left\lbrace F_{1}, ..., F_{T}\right\rbrace$ and event temporal bins $\boldsymbol{S} = \left\lbrace S_{1}, ..., S_{K}\right\rbrace$ are required. Therefore, we keep the frames rate (*i.e.*, 2.5 Hz) at $1/4$ of the event temporal bins frame rate (*i.e.*, depth ground truth frame rate, 10 Hz). The training set consists of 35 sequences with more than 5.4k frames, and 21.9k depth ground-truth and corresponding recorded events. The testing set consists of 6 sequences with more than 1k frames, 4.2k depth ground-truth and corresponding recorded events. During inference, we achieve high-rate monocular depth estimation by increasing the input rate of frames and event temporal bins, especially by increasing the rate of the latter. In addition, we also discuss the effectiveness of CFRNet in handling two modalities with the same frame rate on the MVSEC dataset. For MVSEC, we follow the experimental settings of previous works, training the monocular depth estimation methods on the outdoor day2 sequence and evaluating them on the outdoor day1 and outdoor night1 sequences.

<table class="tg">
			<tr>
				<th class="tg-tpii">Outdoors Day1<br></th>
				<th class="tg-tpii">Outdoors Day2<br></th>
				<th class="tg-tpii">Outdoors Night1<br></th>
				<th class="tg-tpii">Outdoors Night2<br></th>
				<th class="tg-tpii">Outdoors Night3<br></th>
			</tr>
			<tr>
				<td class="tg-qlrr"><a href="[data/E2DEPTH/mvsec/mvsec_outdoor_day1.tar](https://rpg.ifi.uzh.ch/data/E2DEPTH/mvsec/mvsec_outdoor_day1.tar)">outdoor_day1.tar (4.1 GB)</a></td>
				<td class="tg-qlrr"><a href="data/E2DEPTH/mvsec/mvsec_outdoor_day2.tar">outdoor_day2.tar (7.9 GB)</a></td>
				<td class="tg-qlrr"><a href="data/E2DEPTH/mvsec/mvsec_outdoor_night1.tar">outdoor_night1.zip (3.3 GB)</a></td>
				<td class="tg-qlrr"><a href="data/E2DEPTH/mvsec/mvsec_outdoor_night2.tar">outdoor_night2.zip (3.4 GB)</a></td>
				<td class="tg-qlrr"><a href="data/E2DEPTH/mvsec/mvsec_outdoor_night3.tar">outdoor_night3.zip (3.0 GB)</a></td>
</tr>
</table>

High-rate monocular depth estimation at 100Hz, where the frame modality frame rate is 2.5Hz and the event modality is 100Hz:

![High-rate monocular depth estimation at 100Hz](https://github.com/liuxu0303/CFRNet/blob/main/High_rate_depth_100Hz.gif)
