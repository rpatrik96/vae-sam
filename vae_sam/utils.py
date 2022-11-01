def add_tags(args):
    try:
        args.tags
    except:
        args.tags = []

    if args.tags is None:
        args.tags = []

    if args.model.prior == "uniform":
        args.tags.append("uniform_prior")
    elif args.model.prior == "gaussian":
        args.tags.append("gaussian_prior")
    elif args.model.prior == "beta":
        args.tags.append("beta_prior")
    elif args.model.prior == "laplace":
        args.tags.append("laplace_prior")

    return list(set(args.tags))
