# Decanus to Legatus: Synthetic training for 2D-3D human pose lifting

This is the original implementation of the Paper "Decanus to Legatus: Synthetic training for 2D-3D human pose lifting"

## Required packages:

* pytorch
* Pillow (PIL)
* matplotlib
* numpy

## Sample Code

* To reproduce a handcrafted distribution, run:

```
python train.py --task make-distribution
```
or

```
python train.py --task md
```

The reproduced distribution will be saved inside './distribution/handcrafted' folder.

* To draw generation samples from our handcrafted distribution, run:

```
python train.py --task generation
```
or

```
python train.py --task g
```

The samples will be saved inside './examples' folder.

* To train, run:

```
python train.py --task train
```

or

```
python train.py --task t
```

The weights will be saved in 'model' folder under name as 'lifter_(epoch).pth'

* To show inference examples of our pretrained model on several COCO dataset samples, run:

```
python train.py --task inference
```

or

```
python train.py --task i
```

The samples will be saved inside './examples' folder.

* To do all former 4 steps in the introduced order, run:

```
python train.py
```

