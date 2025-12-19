from dataclasses import dataclass, field
from tqdm.auto import tqdm
from .opt_strategy import OptStrategy, OptStrategyModule


@dataclass
class GradAccOptStrategy(OptStrategy):
    from_steps: list[int] = field(default_factory=lambda: [
        0,
        20_000,
        24_000,
    ])

    acc_steps: list[int] = field(default_factory=lambda: [
        1,
        5,
        20,
    ])

    def instantiate(self):
        return GradAccOptStrategyModule(self)


class GradAccOptStrategyModule(OptStrategyModule):
    def _classify_optimizers(self, optimizers):
        if isinstance(optimizers, list) is False:
            optimizers = [optimizers]

        non_sparse = []
        sparse = []

        for opt in optimizers:
            if opt.__module__.startswith("torch.optim."):
                non_sparse.append(opt)
            else:
                sparse.append(opt)

        return non_sparse, sparse

    def _process_lr_schedulers(self, schedulers):
        if schedulers is None:
            schedulers = []
        if isinstance(schedulers, list) is False:
            schedulers = [schedulers]

        return schedulers

    def on_train_start(self, pl_modules):
        self.optimizers, self.sparse_adams = self._classify_optimizers(pl_modules.unwrapped_optimizers)
        tqdm.write("{}/{} SparseAdam".format(len(self.sparse_adams), len(self.optimizers) + len(self.sparse_adams)))

        schedulers = pl_modules.unwrapped_schedulers
        if schedulers is None:
            schedulers = []
        if isinstance(schedulers, list) is False:
            schedulers = [schedulers]
        self.schedulers = schedulers

        self.from_steps = list(self.config.from_steps) + [9999999]
        self.acc_steps = list(self.config.acc_steps) + [self.config.acc_steps[-1]]

        self.cur_acc = 0
        global_step = pl_modules.trainer.global_step
        for idx, step in enumerate(self.from_steps):
            if global_step > step:
                self.cur_acc = idx
        tqdm.write("acc_steps={}".format(self.acc_steps[self.cur_acc]))

    def step(self, global_step, pl_module):
        if global_step % self.acc_steps[self.cur_acc] == 0:
            for opt in self.optimizers:
                opt.step()
                opt.zero_grad(set_to_none=True)

        for opt in self.sparse_adams:
            opt.step()
            opt.zero_grad(set_to_none=True)

        for scheduler in self.schedulers:
            scheduler.step()

        if global_step > self.from_steps[self.cur_acc + 1]:
            self.cur_acc += 1
            tqdm.write("acc_steps={}".format(self.acc_steps[self.cur_acc]))

        pl_module.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.total.completed += 1
