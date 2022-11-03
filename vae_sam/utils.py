from os.path import dirname, abspath, join


DATA_DIR = join(dirname(dirname(abspath(__file__))), "data")
CIFAR10_DIR = join(DATA_DIR, "cifar10")


def add_tags(args):
    try:
        args.tags
    except:
        args.tags = []

    if args.tags is None:
        args.tags = []

    if args.model.sam_update is True:
        args.tags.append("sam")

    return list(set(args.tags))
