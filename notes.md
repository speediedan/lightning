Trainer(resume_from_checkpoint="path")


- When do we restore what?
- How do we let users customize what to restore?


Which objects can have state in Lightning?
- Trainer
- LightningModule
- Callbacks
- Plugins? Loggers?


Current sequence:

fit:
- prepare_data(model)
- attach_model_callbacks(model)
- on_before_accelerator_backend_setup(model)
- accelerator.connect(model)
- accelerator.setup_environment()
  restore_weights()
- call_setup_hook(model)
- call_configure_sharded_model(model)
  restore trainer state()
- accelerator.setup(model)  <-- optimizer setup and assigned

- run_train()
    - pre_training_routine()
        - restore_weights()

            # the magic
            - training_type_plugin.restore_model_state_from_ckpt_path(checkpoint_path)  -> calls model.on_load_checkpoint()
            - model.cuda()
            - restore_training_state(checkpoint)
                - amp state
                - Callback.on_load_checkpoint(checkpoint)
                - global_step, epoch
                - optimizer state restore
                - lr_scheduler restore





Related issues:
https://github.com/PyTorchLightning/pytorch-lightning/issues/7535#issuecomment-842508795


```python

class ToyTask(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()


        # hidden
        self.optimizers
        self.schedulers

        # method overwrite or property??
        model.optimizers = [...]

        # user
        self.optimizers = ..

    def setup(self, stage: str):
        if stage == "test":
            return

        self.model = ToyModel()
        self.optimizer = AdamW(self.model.parameters(), lr=0.001, betas=[0.9, 0.999], eps=1.0e-08, weight_decay=0, amsgrad=False)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        targets = self.forward(batch["model_input"])
        loss = self.loss_fn(targets, batch["label"])

        # Log loss results per train step and per epoch
        self.log("loss", loss)

        # Tell Lightning to minimize loss

        opt = self.optimizers() -> now: trainer.accleerator.optimizers

        return loss

    def configure_optimizers(self):
        return Adam()
        return self.optimizer

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.setup("fit")
        self.trainer.accelerator.setup(trainer, self)


# method overwrite
model.optimizers = [...]

trainer = pl.Trainer(
    gpus=1,
    precision=16,
    callbacks=[model_checkpoint],
)

trainer.fit(task, train_dataloader)

trainer = pl.Trainer(
    gpus=1,
    precision=16,
    callbacks=[model_checkpoint],
    resume_from_checkpoint=model_checkpoint.last_model_path,
)
trainer.fit(task, train_dataloader) <--- this is where will fail

```


Make optimizer
