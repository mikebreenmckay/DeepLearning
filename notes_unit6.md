## Unit 6 Deep Learning Tips and Tricks

- Replace some of the basic setting and choices around optimization algorithms and activation functions leveraging features in lightning
- Learn about systematic ways to tune neural networks

### Model Checkpointing/Early Stopping

- Motivation
    - models may become worse over time due to overfitting (training loss continues to decrease faster than validation loss)
    - Not the case that the last epoch will be the most accurate wrt validation data

If we see that the second to last epoch was better we could rerun the model training with one fewer epochs: **Early Stopping**

With model checking we can save the best checkpoints:
```py
from lightning.pytorch.callbacks import ModelCheckpoint

callbacks = [
    ModelCheckpoint(save_top_k=k, mode="max", monitor="val_acc", save_last=True)
]
```
where k is the number of best checkpoints we want to save. 

We can then evaluate the model on the best checkpoint:
```py
trainer.test(model=lightning_model, datamodule=dm, ckpt_path="best")
```



