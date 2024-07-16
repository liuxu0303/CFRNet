## High-rate monocular depth estimation via cross frame-rate collaboration of frames and events

![avatar](Pictures/framework.jpg)

The pipeline of the proposed high-rate monocular depth estimator using a *cross frame-rate frame-event joint learning network (**CFRNet**)*. The continuous event stream is first split into high-rate event temporal bins and converted into event representations. For the current timestamp $t_i$, we use the modal-specific shared encoder adopting a lightweight CNN-Transformer hybrid backbone (*i.e.,* lite-mono) to extract local-global features from the event representation $E_i$ and the most recent frame $F_j$, respectively. Then, the proposed CFMF utilizes implicit spatial alignment and dynamic attention-based fusion strategies to generate a complementary joint representation. Meanwhile, a novel  recurrent network TCM effectively models long-range temporal dependencies between the joint representations. Finally, a normal CNN-based decoder predicts high-rate and fine-grained depth maps.

High-rate monocular depth estimation at 100Hz, where the frame modality frame rate is 2.5Hz and the event modality is 100Hz:

![High-rate monocular depth estimation at 100Hz](https://github.com/liuxu0303/CFRNet/blob/main/High_rate_depth_100Hz.gif)
