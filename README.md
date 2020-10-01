# Deep Active Countor Model(ACM) for Buildings Segmentations
<p align="center">
  <img src="pic/sample01.jpg" width="800">
</p>
 Deep snake algorithm for 2D images based on - [ICLR2020 paper(revisiting the results)](https://arxiv.org/abs/1912.00367)
 Architecture based on [Hardnet85](https://arxiv.org/abs/1909.00948)

### Get Started
**To train a segmentation model :**
```
python train_seg.py -bs 50 -WD 0.00005 -D_rate 3000 -task bing -opt sgd -lr 0.02 -nW 8
```
**To train a ACM model :**
```
python train.py -bs 25 -WD 0.00005 -D_rate 30 -it 2
```
**To eval a segmentation model :**
```
python eval.py -task viah -nP 100 -it 3 -a 0.4
```
**To eval a ACM model :**
```
 python eval_seg.py -task bing -nP 24 -it 2 -a 0.4
```
### Results  
  
<p align="center">
  <img src="pic/fchardnet70_cityscapes.png" width="420" title="Cityscapes">
</p>  

| Method | Viah <br> mIoU  | Bing <br> mIoU| 
| :---: |  :---:  |  :---:  | 
| DARNet  | 88.24  | 75.29  | 
| DSAC | 71.10 | 38.74 | 
| **ours** | **90.33**  | **75.53** | 
