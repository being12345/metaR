import torch
import argparse


def get_params():
    args = argparse.ArgumentParser()

    # data
    args.add_argument("-data", "--dataset", default="NELL-One", type=str)  # ["NELL-One", "Wiki-One"]
    args.add_argument("-path", "--data_path", default="./NELL", type=str)  # ["./NELL", "./Wiki"]
    args.add_argument("-form", "--data_form", default="Pre-Train", type=str)  # ["Pre-Train", "In-Train", "Discard"]
    args.add_argument("-seed", "--seed", default=42, type=int)
    args.add_argument("-few", "--few", default=3, type=int)
    args.add_argument("-nq", "--num_query", default=3, type=int)
    args.add_argument("-bfew", "--base_classes_few", default=3, type=int)
    args.add_argument("-bnq", "--base_classes_num_query", default=3, type=int)
    args.add_argument("-br", "--base_classes_relation", default=30, type=int)
    args.add_argument("-metric", "--metric", default="MRR", choices=["MRR", "Hits@10", "Hits@5", "Hits@1"])
    # metaR model
    args.add_argument("-dim", "--embed_dim", default=100, type=int)
    args.add_argument("-b", "--beta", default=5, type=float)
    args.add_argument("-m", "--margin", default=1, type=float)
    args.add_argument("-p", "--dropout_p", default=0.5, type=float)
    args.add_argument("-abla", "--ablation", default=False, type=bool)
    # VAE model
    args.add_argument("--hidden_size", type=int, default=100, help="hidden size of transformer model")
    args.add_argument("--num_hidden_layers", type=int, default=1, help="number of layers")
    args.add_argument('--num_attention_heads', default=4, type=int)
    args.add_argument('--hidden_act', default="gelu", type=str)  # gelu relu
    args.add_argument("--attention_probs_dropout_prob", type=float, default=0.0, help="attention dropout p")
    args.add_argument("--hidden_dropout_prob", type=float, default=0.3, help="hidden dropout p")
    args.add_argument("--initializer_range", type=float, default=0.02)
    args.add_argument('--max_seq_length', default=1, type=int)
    # VAE variants
    args.add_argument("--reparam_dropout_rate", type=float, default=0.2,
                      help="dropout rate for reparameterization dropout")
    args.add_argument("--latent_clr_weight", type=float, default=0.3, help="weight for latent clr loss")

    # contrastive loss
    args.add_argument('--temperature', type=float, default=0.5)
    # KL annealing args
    args.add_argument('--anneal_cap', type=float, default=0.3)
    args.add_argument('--total_annealing_step', type=int, default=10000)
    # train
    args.add_argument("-bs", "--batch_size", default=3, type=int)
    args.add_argument("-nt", "--num_tasks", default=8, type=int)
    args.add_argument("-lr", "--learning_rate", default=0.001, type=float)
    args.add_argument("-es_p", "--early_stopping_patience", default=30, type=int)
    args.add_argument("-epo", "--epoch", default=1500, type=int)
    args.add_argument("-bepo", "--base_epoch", default=5500, type=int)
    args.add_argument("-prt_epo", "--print_epoch", default=50, type=int)
    args.add_argument("-eval_epo", "--eval_epoch", default=1499, type=int)
    args.add_argument("-beval_epo", "--base_eval_epoch", default=5449, type=int)
    args.add_argument("-ckpt_epo", "--checkpoint_epoch", default=1000, type=int)
    args.add_argument("-gpu", "--device", default=0, type=int)

    args.add_argument("-prefix", "--prefix", default="exp1", type=str)
    args.add_argument("-step", "--step", default="train", type=str, choices=['train', 'test', 'dev'])
    args.add_argument("-log_dir", "--log_dir", default="log", type=str)
    args.add_argument("-state_dir", "--state_dir", default="state", type=str)
    args.add_argument("-eval_ckpt", "--eval_ckpt", default=None, type=str)
    args.add_argument("-eval_by_rel", "--eval_by_rel", default=False, type=bool)

    args = args.parse_args()
    args.device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    params = {}
    for k, v in vars(args).items():
        params[k] = v

    if args.dataset == 'NELL-One':
        params['embed_dim'] = 100
    elif args.dataset == 'Wiki-One':
        params['embed_dim'] = 50

    return params, args


data_dir = {
    'train_tasks_in_train': '/train_tasks_in_train.json',
    'train_tasks': '/continual_train_tasks.json',
    'test_tasks': '/test_tasks.json',
    'dev_tasks': '/continual_dev_tasks.json',
    'few_shot_dev_tasks': '/dev_tasks.json',

    'rel2candidates_in_train': '/rel2candidates_in_train.json',
    'rel2candidates': '/rel2candidates.json',

    'e1rel_e2_in_train': '/e1rel_e2_in_train.json',
    'e1rel_e2': '/e1rel_e2.json',

    'ent2ids': '/ent2ids',
    'ent2vec': '/ent2vec.npy',
}
