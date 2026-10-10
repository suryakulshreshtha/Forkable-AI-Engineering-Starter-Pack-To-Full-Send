# Tiny Autograd

Backpropagation is bookkeeping: every number remembers which numbers made it, and the chain rule walks that record backwards. You build the record keeper, stack it into neurons and layers, and train a network on XOR. When it learns, you will know there was never any magic, only a lot of small multiplications in the right order.
