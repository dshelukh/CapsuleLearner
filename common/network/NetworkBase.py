'''
@author: Dmitry
'''

class BasicNet():
    def __init__(self, modes_dict, mode, elements_dict = None, config = None):
        self.modes = modes_dict
        self.config = config
        self.runner = None

        self.mode = None
        self.set_mode(mode)
        self.run_elements = None
        self.set_elements(elements_dict)

    def set_mode(self, mode):
        self.mode = mode
        self.runner = self.modes[self.mode]

    def set_elements(self, elements_dict):
        self.run_elements = elements_dict

    def run(self, *args, **kwargs):
        print('Run:', self.mode)
        return self.runner.run(self.config, self.run_elements, *args, **kwargs)

    def loss_function(self, *args, **kwargs):
        print('Loss:', self.mode)
        return self.runner.loss_function(self.config, *args, **kwargs)

    def num_classified(self, *args, **kwargs):
        return self.runner.num_classified(self.config, *args, **kwargs)

    def get_minimizers(self, *args, **kwargs):
        return self.runner.get_minimizers(*args, **kwargs)

    def get_functions_for_trainer(self):
        return 'loss_function', 'run', 'num_classified', 'get_minimizers'

    def get_runner(self, mode = None):
        return self.runner if mode is None else self.modes[mode]



