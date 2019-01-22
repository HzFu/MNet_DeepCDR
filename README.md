# MNet_CDR_Seg

**Code for TMI 2018 "Joint Optic Disc and Cup Segmentation Based on Multi-label Deep Network and Polar Transformation"**

Project homepage：http://hzfu.github.io/proj_glaucoma_fundus.html

1. The code is based on: *Keras 2.0 + Tensorflow 1.0 + Matlab*
2. The deep output is raw segmentation result without ellipse fitting. The Matlab code is the ellipse fitting and CDR calculation (by using PDollar toolbox: https://pdollar.github.io/toolbox/).
3. You can run the 'Step\_3\_MNet\_test.py' for testing any new image directly.
4. We also provided the validation and test results on [REFUGE dataset](https://refuge.grand-challenge.org/home/) in 'REFUGE\_result' fold.


----------------

**Main files:**

1. 'Step\_1\_Disc\_Crop.py': The disc detection code for whole funuds image.
2. 'Step\_2\_MNet\_train.py': The M-Net training code.
3. 'Step\_3\_MNet\_test.py': The M-Net testing code.
4. 'Step\_4\_CDR\_output.m': The ellipse fitting for disc and cup, and CDR calculation.

----------------
**If you use this code, please cite the following papers:**

[1] Huazhu Fu, Jun Cheng, Yanwu Xu, Damon Wing Kee Wong, Jiang Liu, and Xiaochun Cao, "Joint Optic Disc and Cup Segmentation Based on Multi-label Deep Network and Polar Transformation", IEEE Transactions on Medical Imaging (TMI), vol. 37, no. 7, pp. 1597–1605, 2018. ([ArXiv version](https://arxiv.org/abs/1801.00926))  

[2] Huazhu Fu, Jun Cheng, Yanwu Xu, Changqing Zhang, Damon Wing Kee Wong, Jiang Liu, and Xiaochun Cao, "Disc-aware Ensemble Network for Glaucoma Screening from Fundus Image", IEEE Transactions on Medical Imaging (TMI), vol. 37, no. 11, pp. 2493–2501, 2018. ([ArXiv version](http://arxiv.org/abs/1805.07549))


----------------
**Note: for ORIGA and SCES datasets**

Unfortunately, the ORIGA and SCES datasets cannot be released due to the clinical policy.
But, here is an other glaucoma challenge, [**Retinal Fundus Glaucoma Challenge (REFUGE)**](https://refuge.grand-challenge.org/home/), including disc/cup segmentation, glaucoma screening, and localization of Fovea. 

----------------
Update log:

- 19.01.22: Added training code, and uploaded the results on REFUGE dataset.
- 18.06.30: Added ellipse fitting code (based on Matlab), and Fixed the bug for macular center fundus.
- 18.06.29: Added disc detection code (based on U-Net).
- 18.02.26: Added CDR calculation code (based on Matlab).
- 18.02.24: Released the code.

