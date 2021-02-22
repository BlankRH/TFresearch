## SETTING

- tensorflow 2.3.2
- cuda 10.1
- tensorflow-determinism
- win10
- GTX1060 3G



- Model: Alexnet (vgg16 exceeded memory limit)
- Dataset: cifar10 (size reduced)

## TECHNIQUES

- enable nvidia deterministic settings

```python
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
```

- initialize random seed

```python
SEED = 123
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
```

- initialize Conv2D and Dense layers with seed

```python
Conv2D(..., kernel_initializer=glorot_normal(seed=SEED))
Dense(..., kernel_initializer=glorot_normal(seed=SEED))
```

- disable shuffle and parallelism in model.fit

```python
model.fit(..., workers=1, shuffle=False)
```



## RESULT

### NON-DETERMINISM

- accuracy

![n_acc](G:\TFresearch\experiment\n_acc.png)

- loss

![](G:\TFresearch\experiment\n_loss.png)

- converge at around 100 epochs
- training result varies

### Determinism

- accuracy

![](G:\TFresearch\experiment\d_acc.png)

- loss

![](G:\TFresearch\experiment\d_loss.png)

- All identical
- converges earlier than non-deterministic training
- higher loss (may be affected by the training data size)

### Weight Comparison

![](G:\TFresearch\experiment\py.JPG)

- Determinism training results in completely identical weights