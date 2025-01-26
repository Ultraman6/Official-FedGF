import argparse
MODELS = ['FedSAMcnn', 'resnet18_nonorm']

def get_parser():
    parser = argparse.ArgumentParser(description="Standalone training")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--model", default='FedSAMcnn', type=str, choices=MODELS)
    parser.add_argument("--pre_trained", action='store_true')
    parser.add_argument("--alg", default='fedsam', type=str)
    parser.add_argument("--dataset", default='cifar10', type=str)
    parser.add_argument("--eval_every", default=800, type=int)
    parser.add_argument("--avg_test", default=True, action='store_true')
    parser.add_argument("--save_model", default=False, action='store_true')

    # dataset distribution
    parser.add_argument("--balance", default=True, action='store_true')
    parser.add_argument("--partition", default='dirichlet', type=str)
    parser.add_argument("--dir_alpha", default=0.5, type=str)
    parser.add_argument("--num_shards", default=10, type=int)
    parser.add_argument("--transform", default=True, action='store_true')

    parser.add_argument("--wandb_project_name", default='test for fednfa async', type=str)
    parser.add_argument('--exp_name', default='fednfa', action='store_true')
    parser.add_argument("--total_client", type=int, default=100)
    parser.add_argument("--com_round", default=10000, type=int)

    parser.add_argument("--sample_ratio", default=0.1, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--weight_decay", type=float, default=0.0004)
    parser.add_argument("--momentum", type=float, default=0)

    # scaffold
    parser.add_argument("--g_lr", type=float)
    # feddyn
    parser.add_argument("--alpha", type=float)
    # fedprox
    parser.add_argument("--mu", type=float)
    # fedsam
    parser.add_argument("--rho", default=0.02, type=float)
    # fedasam
    parser.add_argument("--eta", type=float)
    # mofedsam, fedavgm, nagfedsam
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--eta_g", type=float, default=1)
    # FedGF, nagfedsam
    parser.add_argument("--T_D", type=float, default=0.2)
    parser.add_argument("--g_rho", type=float, default=0.02)
    parser.add_argument("--W", type=int, default=10)
    # FedLESAM
    parser.add_argument("-isLocal", type=bool, default=False)
    parser.add_argument("-isNAG", type=bool, default=False)
    parser.add_argument("-isCum", type=bool, default=False)
    parser.add_argument("-noTrain", type=bool, default=False)

    args = parser.parse_args()
    if args.partition == 'iid':
        dp = args.partition
    elif args.partition == 'dirichlet':
        dp = f"{args.partition}_dir_alpha:{args.dir_alpha}"
    elif args.partition == 'shards':
        dp = f"{args.partition}_num_shards:{args.num_shards}"
    else:
        raise ValueError(f"Unknown partition {args.partition}")
    args.exp_name = f"{args.alg}_{args.dataset}_{dp}_rho:{args.rho}_num_clients:{args.total_client}_sample_ratio:{args.sample_ratio}"
    return parser.parse_args()
