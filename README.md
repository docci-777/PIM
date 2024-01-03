# PIM
PERMUTATION-INVARIANT ATTACKS

## Environments
The experiments in the paper are conducted under the following environment:
* Python 3.9.13
* Pytorch 1.12.1
* Torchvision 0.13.1
* Timm 0.9.2
* Numpy 1.21.5
* Einops 0.7.0
  
### Key parameters
```--nn```: Network architecture.

```--data_dir```: Directory of dataset.

```--log_dir```: Directory of log file.

```--eps```: Perturbation size.

```--alpha```: Step size.

```--m1```: Control the number of scales in SI.

```--m2```: Control the number of mixed images in Admix.

### Evaluate PIFGSM
- To evaluate *default* PI:

```
python pifgsm.py --batch_size 32 --log_name PI_MI --m2 0 --m1 0 --diversity_prob 0
```

- To evaluate PI integrate with other methods:

```
python pifgsm.py --batch_size 32 --log_name PI_MASDI
```
