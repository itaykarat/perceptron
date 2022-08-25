class activation_function:


    def __init__(self):
        self.types = ['step function','sigmoid']

    def step_func(self,z):
        return 1.0 if (z > 0) else 0.0