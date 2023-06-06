import argparse
import logging

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def main(args):
    logging.info(args.mode)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=["train", "infer", "eval", "visualize"])
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    main(args)
