import torch

import data
import loss
import model
import utility
from option import args
from trainer import Trainer
from videotester import VideoTester

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)


def main():
    global model
    if args.data_test == ["video"]:

        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
                print(args.model, args.save)
                print("using gpus : {}".format(args.gpus))
                if hasattr(_model.model, "update_epoch"):
                    _model.model.update_epoch(t.optimizer.get_last_epoch(), args.epochs)
                t.train()
                t.test()
            print(args.model, args.save)
            print("using gpus : {}".format(args.gpus))

            checkpoint.done()


if __name__ == "__main__":
    main()
