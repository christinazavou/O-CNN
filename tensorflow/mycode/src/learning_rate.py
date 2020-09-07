import tensorflow as tf


class StepLR:
    def __init__(self, **kwargs):
        self.step_size = kwargs['step_size']
        self.gamma = kwargs['gamma']
        self.learning_rate = kwargs['learning_rate']

    def __call__(self, global_step):
        with tf.variable_scope('step_lr'):
            step_size = list(self.step_size)
            for i in range(len(step_size), 5):
                step_size.append(step_size[-1])

            steps = step_size
            for i in range(1, 5):
                steps[i] = steps[i - 1] + steps[i]
            lr_values = [self.gamma ** i * self.learning_rate for i in range(0, 6)]

            lr = tf.train.piecewise_constant(global_step, steps, lr_values)
        return lr


class LRFactory:
    def __init__(self, **kwargs):
        if kwargs['lr_type'] == 'step':
            self.lr = StepLR(**kwargs)
        else:
            raise Exception('Unsupported learning rate: ' + kwargs['lr_type'])

    def __call__(self, global_step):
        return self.lr(global_step)
