# PIM
PERMUTATION-INVARIANT ATTACKS

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
