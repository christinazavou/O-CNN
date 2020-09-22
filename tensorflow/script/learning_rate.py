import tensorflow as tf


class CosLR:
    def __init__(self, flags):
        self.flags = flags

    def __call__(self, global_step):
        with tf.variable_scope('cos_lr'):
            pi, mul = 3.1415926, 0.001
            step_size = self.flags.step_size[0]
            max_iter = self.flags.max_iter * 0.9
            max_epoch = max_iter / step_size
            lr_max = self.flags.learning_rate
            lr_min = self.flags.learning_rate * mul
            epoch = tf.floordiv(tf.cast(global_step, tf.float32), step_size)
            val = tf.minimum(epoch / max_epoch, 1.0)
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + tf.cos(pi * val))
        return lr


class StepLR:
    def __init__(self, flags):
        self.flags = flags

    def __call__(self, global_step):
        with tf.variable_scope('step_lr'):
            step_size = list(self.flags.step_size)
            for i in range(len(step_size), 5):
                step_size.append(step_size[-1])

            steps = step_size
            for i in range(1, 5):
                steps[i] = steps[i - 1] + steps[i]
            lr_values = [self.flags.gamma ** i * self.flags.learning_rate for i in range(0, 6)]

            lr = tf.train.piecewise_constant(global_step, steps, lr_values)
        return lr


class OnPlateauLR:
    def __init__(self, flags):
        self.flags = flags
        self.best_metric = tf.Variable(initial_value=0.) if self.flags.mode == 'max' \
            else tf.Variable(initial_value=1.e100)
        self.prev_lr = tf.Variable(initial_value=self.flags.learning_rate)
        self.cooldown_counter = tf.Variable(initial_value=0)
        self.bad_epochs = tf.Variable(initial_value=0)

    def is_better(self, curr_metric):
        if self.flags.mode == 'min' and self.flags.threshold_mode == 'rel':
            rel_epsilon = 1. - self.flags.threshold
            return tf.less(curr_metric, tf.multiply(self.best_metric, rel_epsilon))

        elif self.flags.mode == 'min' and self.flags.threshold_mode == 'abs':
            return tf.less(curr_metric, tf.subtract(self.best_metric, self.flags.threshold))

        elif self.flags.mode == 'max' and self.flags.threshold_mode == 'rel':
            rel_epsilon = self.flags.threshold + 1.
            return tf.greater(curr_metric, tf.multiply(self.best_metric, rel_epsilon))

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return tf.greater(curr_metric, tf.add(self.best_metric, self.flags.threshold))

    def __call__(self, curr_metric):
        # new_lr = tf.Variable(initial_value=0)
        with tf.variable_scope('plateau_lr'):
            self.bad_epochs, self.best_metric = tf.cond(
                self.is_better(curr_metric),
                true_fn=lambda: [tf.subtract(self.bad_epochs, self.bad_epochs), curr_metric],
                false_fn=lambda: [tf.add(self.bad_epochs, 1), self.best_metric]
            )

            self.bad_epochs, self.cooldown_counter = tf.cond(
                tf.greater(self.cooldown_counter, 0),
                true_fn=lambda: [0, tf.add(self.cooldown_counter, -1)],
                false_fn=lambda: [self.bad_epochs, self.cooldown_counter]
            )

            new_lr, self.cooldown_counter, self.bad_epochs = tf.cond(
                tf.greater(self.bad_epochs, self.flags.patience),
                true_fn=lambda: [tf.maximum(tf.multiply(self.prev_lr, self.flags.gamma), self.flags.min_lr),
                                 self.flags.cooldown,
                                 0],
                false_fn=lambda: [self.prev_lr, self.cooldown_counter, self.bad_epochs]
            )

            new_lr = tf.cond(tf.greater(tf.subtract(self.prev_lr, new_lr), self.flags.eps),
                             true_fn=lambda: new_lr,
                             false_fn=lambda: self.prev_lr)
            self.prev_lr = new_lr
            return new_lr


class OnPlateauLRPy:
    def __init__(self, flags):
        self.flags = flags
        self.best_metric = 0. if self.flags.mode == 'max' else 1.e100
        self.prev_lr = self.flags.learning_rate
        self.cooldown_counter = 0
        self.bad_epochs = 0

    def is_better(self, curr_metric):
        if self.flags.mode == 'min' and self.flags.threshold_mode == 'rel':
            rel_epsilon = 1. - self.flags.threshold
            return curr_metric < self.best_metric * rel_epsilon

        elif self.flags.mode == 'min' and self.flags.threshold_mode == 'abs':
            return curr_metric < self.best_metric - self.flags.threshold

        elif self.flags.mode == 'max' and self.flags.threshold_mode == 'rel':
            rel_epsilon = self.flags.threshold + 1.
            return curr_metric > self.best_metric * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return curr_metric > self.best_metric + self.flags.threshold

    def __call__(self, curr_metric):
        if self.is_better(curr_metric):
            self.bad_epochs = 0
            self.best_metric = curr_metric
        else:
            self.bad_epochs += 1

        if self.cooldown_counter > 0:
            self.bad_epochs = 0
            self.cooldown_counter -= 1

        if self.bad_epochs > self.flags.patience:
            new_lr = max(self.prev_lr * self.flags.gamma, self.flags.min_lr)
            self.cooldown_counter = self.flags.cooldown
            self.bad_epochs = 0
        else:
            new_lr = self.prev_lr

        if self.prev_lr - new_lr <= self.flags.eps:
            new_lr = self.prev_lr

        self.prev_lr = new_lr
        return new_lr


class LRFactory:
    def __init__(self, flags):
        self.flags = flags
        if self.flags.lr_type == 'step':
            self.lr = StepLR(flags)
        elif self.flags.lr_type == 'cos':
            self.lr = CosLR(flags)
        elif self.flags.lr_type == 'plateau':
            self.lr = OnPlateauLRPy(flags)
        else:
            print('Error, unsupported learning rate: ' + self.flags.lr_type)

    def __call__(self, global_step):
        return self.lr(global_step)
