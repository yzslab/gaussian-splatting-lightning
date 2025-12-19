from dataclasses import dataclass
from .opt_strategy import OptStrategy, OptStrategyModule


@dataclass
class VanillaOptStrategy(OptStrategy):
    def instantiate(self):
        return VanillaOptStrategyModule(self)


class VanillaOptStrategyModule(OptStrategyModule):
    @staticmethod
    def do_nothing():
        return

    def _multiple_optimizer_step_fix(self, optimizers):
        if isinstance(optimizers, list) is False:
            return [optimizers]

        """
        IMPORTANCE: the global_step will be increased on every step() call of all the optimizers,
        issue https://github.com/Lightning-AI/lightning/issues/17958,
        here change _on_before_step and _on_after_step to override this behavior.
        """
        for idx, optimizer in enumerate(optimizers):
            if idx == 0:
                continue
            optimizer._on_before_step = self.do_nothing
            optimizer._on_after_step = self.do_nothing

        return optimizers

    def _process_lr_schedulers(self, schedulers):
        if schedulers is None:
            schedulers = []
        if isinstance(schedulers, list) is False:
            schedulers = [schedulers]

        return schedulers

    def step(self, global_step, pl_module):
        for opt in self._multiple_optimizer_step_fix(pl_module.optimizers()):
            opt.step()
            opt.zero_grad(set_to_none=True)

        for scheduler in self._process_lr_schedulers(pl_module.lr_schedulers()):
            scheduler.step()
