def classic(lr, warm_up = 3, decay = 0.9):

    def lr_schedule():
        # Intial value
        factor = lr * 1 / 2 ** warm_up
        yield factor
        
        # Multiply with 2 first few round
        for i in range(warm_up):
            factor *= 2
            yield factor

        # Make an exponential decay
        while True:
            factor *= decay
            yield factor

    return lr_schedule