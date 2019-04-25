from subprocess import call


class LrFinder(object):
    def __init__(self, 
                 start_lr=1e-5, 
                 end_lr=1e-3, 
                 num_it=5,
                 epochs=150,
                 load=False):
        
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.num_it = num_it
        self.epochs = epochs
        self.load = load

        self.lr_mult = (float(end_lr) / float(start_lr)) ** (1.0 / float(num_it))
        self.lr = self.start_lr

    def run(self):
        for _ in range(self.num_it):
            if self.load:
                cmd = "python predictor.py --load --iter={0} --lr={1} --epoch={2} --model_name='lr_{1}'".format(
                    0, self.lr, self.epochs
                )
            else:
                cmd = "python predictor.py --iter={0} --lr={1} --epoch={2} --model_name='lr_{1}'".format(
                    0, self.lr, self.epochs
                )
            call(cmd, shell=True)
            self.lr *= self.lr_mult


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--start_lr', default=1e-5, type=float)
    parser.add_argument('--end_lr', default=1e-2, type=float)
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--num_it', default=10, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    args = parser.parse_args()
    finder = LrFinder(
        start_lr=args.start_lr,
        end_lr=args.end_lr,
        num_it=args.num_it,
        epochs=args.epochs,
        load=args.load)
        
    finder.run()
