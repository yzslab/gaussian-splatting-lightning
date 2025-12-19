class OptStrategy:
    def instantiate(self):
        raise NotImplementedError()


class OptStrategyModule:
    def __init__(self, config):
        self.config = config

    def on_train_start(self, pl_modules):
        return

    def step(self):
        raise NotImplementedError()
