import argparse

import torch
import torch.nn as nn


class NCELoss(nn.Module):
    def __init__(self, temperature, device):
        super(NCELoss, self).__init__()
        self.device = device
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.temperature = temperature
        self.cossim = nn.CosineSimilarity(dim=-1).to(self.device)

    def forward(self, batch_sample_one, batch_sample_two):  # batch_size*

        sim11 = self.cossim(batch_sample_one.unsqueeze(-2), batch_sample_one.unsqueeze(-3)) / self.temperature
        sim22 = self.cossim(batch_sample_two.unsqueeze(-2), batch_sample_two.unsqueeze(-3)) / self.temperature
        sim12 = self.cossim(batch_sample_one.unsqueeze(-2), batch_sample_two.unsqueeze(-3)) / self.temperature

        d = sim12.shape[-1]
        sim11[..., range(d), range(d)] = float('-inf')
        sim22[..., range(d), range(d)] = float('-inf')
        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        logits = torch.cat([raw_scores1, raw_scores2], dim=-2)
        labels = torch.arange(2 * d, dtype=torch.long, device=logits.device)
        nce_loss = self.criterion(logits, labels)
        return nce_loss


class ContrastVAELoss:
    def __init__(self, args):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cl_criterion = NCELoss(args.temperature, device)
        self.args = args

    def kl_anneal_function(self, anneal_cap, step, total_annealing_step):
        """
        :param step: increment by 1 for every  forward-backward step
        :param k: temperature for logistic annealing
        :param x0: pre-fixed parameter control the speed of anealing. total annealing steps
        :return:
        """
        return min(anneal_cap, (1.0 * step) / total_annealing_step)

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size)  # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()  # [batch*seq_len]
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        auc = torch.sum(
            ((torch.sign(pos_logits - neg_logits) + 1) / 2) * istarget
        ) / torch.sum(istarget)

        return loss, auc

    def loss_fn_latent_clr(self, reconstructed_seq1, reconstructed_seq2, mu1, mu2, log_var1, log_var2, z1, z2,
                           target_pos_seq, target_neg_seq, step):
        """
        compute kl divergence, reconstruction loss and contrastive loss
        :param sequence_reconstructed: b*max_Sq*d
        :param mu: b*d
        :param log_var: b*d
        :param target_pos_seq: b*max_Sq*d
        :param target_neg_seq: b*max_Sq*d
        :return:
        """

        """compute KL divergence"""

        kld_loss1 = torch.mean(-0.5 * torch.sum(1 + log_var1 - mu1 ** 2 - log_var1.exp(), dim=-1))
        kld_loss2 = torch.mean(-0.5 * torch.sum(1 + log_var2 - mu2 ** 2 - log_var2.exp(), dim=-1))
        kld_weight = self.kl_anneal_function(self.args.anneal_cap, step, self.args.total_annealing_step)

        """compute reconstruction loss from Trainer"""
        recons_loss1, recons_auc = self.cross_entropy(reconstructed_seq1, target_pos_seq, target_neg_seq)  # TODO:
        recons_loss2, recons_auc = self.cross_entropy(reconstructed_seq2, target_pos_seq, target_neg_seq)

        """compute clr loss"""
        user_representation1 = torch.sum(z1, dim=1)
        user_representation2 = torch.sum(z2, dim=1)

        contrastive_loss = self.cl_criterion(user_representation1, user_representation2)

        loss = recons_loss1 + recons_loss2 + kld_weight * (
                kld_loss1 + kld_loss2) + self.args.latent_clr_weight * contrastive_loss
        return loss, recons_auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument("--hidden_size", type=int, default=200, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=1, help="number of layers")
    parser.add_argument('--num_attention_heads', default=4, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str)  # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.0, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_seq_length', default=1, type=int)
    # model variants
    parser.add_argument("--reparam_dropout_rate", type=float, default=0.2,
                        help="dropout rate for reparameterization dropout")
    # contrastive loss
    parser.add_argument('--temperature', type=float, default=0.5)
    # KL annealing args
    parser.add_argument('--anneal_cap', type=float, default=0.3)
    parser.add_argument('--total_annealing_step', type=int, default=10000)
    args = parser.parse_args()
    loss = ContrastVAELoss(args)
    print(loss)
