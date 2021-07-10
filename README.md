mnet_deep_cdr
=============
![Python version range](https://img.shields.io/badge/python-2.7%E2%80%933.6+-blue.svg)
**Code for TMI 2018 "Joint Optic Disc and Cup Segmentation Based on Multi-label Deep Network and Polar Transformation"**

Project homepage：http://hzfu.github.io/proj_glaucoma_fundus.html

## Install dependencies

    pip install -r requirements.txt

## Install package

    pip install .

OpenCV will need to be installed separately.

---

1. The code is based on: *TensorFlow 1.14 (with Keras) + Matlab*
2. The deep output is raw segmentation result without ellipse fitting. The Matlab code is the ellipse fitting and CDR calculation (by using PDollar toolbox: https://pdollar.github.io/toolbox/).
3. You can run the 'Step\_3\_MNet\_test.py' for testing any new image directly.
4. We also provided the validation and test results on [REFUGE dataset](https://refuge.grand-challenge.org/home/) in 'REFUGE\_result' fold.
5. **Note: Due to the 'scipy.misc.imresize' in SciPy 1.0.0 has been removed in SciPy 1.3.0, the original trained model 'Model\_MNet\_REFUGE.h5' is not suitable. If you want to segment disc/cup from fundus image, you can consider our newest methods: CE-Net and AG-Net, which obtain the better performances and are also released in:**
	- CE-Net: [https://github.com/Guzaiwang/CE-Net](https://github.com/Guzaiwang/CE-Net) 
	- AG-Net: [https://github.com/HzFu/AGNet](https://github.com/HzFu/AGNet)
6. A pytorch implementation of M-Net could be found in **AG-Net**: [https://github.com/HzFu/AGNet](https://github.com/HzFu/AGNet)


---

**Main files:**

1. 'Step\_1\_Disc\_Crop.py': The disc detection code for whole funuds image.
2. 'Step\_2\_MNet\_train.py': The M-Net training code.
3. 'Step\_3\_MNet\_test.py': The M-Net testing code.
4. 'Step\_4\_CDR\_output.m': The ellipse fitting for disc and cup, and CDR calculation.

---

**If you use this code, please cite the following papers:**

1. Huazhu Fu, Jun Cheng, Yanwu Xu, Damon Wing Kee Wong, Jiang Liu, and Xiaochun Cao, "Joint Optic Disc and Cup Segmentation Based on Multi-label Deep Network and Polar Transformation", IEEE Transactions on Medical Imaging (TMI), vol. 37, no. 7, pp. 1597–1605, 2018. [[PDF]](https://arxiv.org/abs/1801.00926)  
2. Huazhu Fu, Jun Cheng, Yanwu Xu, Changqing Zhang, Damon Wing Kee Wong, Jiang Liu, and Xiaochun Cao, "Disc-aware Ensemble Network for Glaucoma Screening from Fundus Image", IEEE Transactions on Medical Imaging (TMI), vol. 37, no. 11, pp. 2493–2501, 2018. [[PDF]](http://arxiv.org/abs/1805.07549)


**There are also some related works for medical image segmentation for your reference:**

1. "Attention Guided Network for Retinal Image Segmentation," in MICCAI, 2019. [[PDF]](http://arxiv.org/abs/1907.12930) [[Github Code]](https://github.com/Guzaiwang/CE-Net)
2. “CE-Net: Context Encoder Network for 2D Medical Image Segmentation,” IEEE TMI, 2019. [[PDF]](https://arxiv.org/abs/1903.02740) [[Github Code]](https://github.com/HzFu/AGNet)

---

**Note: for ORIGA and SCES datasets**

Unfortunately, the ORIGA and SCES datasets cannot be released due to the clinical policy.
But, here is an other glaucoma challenge, [**Retinal Fundus Glaucoma Challenge (REFUGE)**](https://refuge.grand-challenge.org/home/), including disc/cup segmentation, glaucoma screening, and localization of Fovea. 

---

Update log:

- 19.01.22: Added training code, and uploaded the results on REFUGE dataset.
- 18.06.30: Added ellipse fitting code (based on Matlab), and Fixed the bug for macular center fundus.
- 18.06.29: Added disc detection code (based on U-Net).
- 18.02.26: Added CDR calculation code (based on Matlab).
- 18.02.24: Released the code.
