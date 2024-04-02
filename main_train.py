import argparse

from trainers.trainer import Trainer
from utils.default_args import add_default_args, parse_args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_default_args(parser, 'face_detect', 64, datasetname='train_small_faces')
    args = parse_args(parser)

    trainer = Trainer(args)
    trainer.initialize_model()
    trainer.train(True)
