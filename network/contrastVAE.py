import argparse
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfAttention(nn.Module):
    def __init__(self, args):
        super(SelfAttention, self).__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.hidden_size, args.num_attention_heads))
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)

        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class Intermediate(nn.Module):
    def __init__(self, args):
        super(Intermediate, self).__init__()
        self.dense_1 = nn.Linear(args.hidden_size, args.hidden_size * 4)
        if isinstance(args.hidden_act, str):
            self.intermediate_act_fn = F.relu  # TODO: use gelu
        else:
            self.intermediate_act_fn = args.hidden_act

        self.dense_2 = nn.Linear(args.hidden_size * 4, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, input_tensor):

        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class Layer(nn.Module):  # attention block
    def __init__(self, args):
        super(Layer, self).__init__()
        self.attention = SelfAttention(args)
        self.intermediate = Intermediate(args)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        return intermediate_output


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        layer = Layer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        """

        :param hidden_states: bxmax_Sqxd
        :param attention_mask: b*1*max_Sq*max_Sq
        :param output_all_encoded_layers: True or False
        :return:
        """

        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        layer = Layer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        """

        :param hidden_states: bxmax_Sqxd
        :param attention_mask: b*1*max_Sq*max_Sq
        :param output_all_encoded_layers: True or False
        :return:
        """

        all_decoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_decoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_decoder_layers.append(hidden_states)
        return all_decoder_layers


class ContrastVAE(nn.Module):

    def __init__(self, args):
        super(ContrastVAE, self).__init__()
        self.item_encoder_mu = Encoder(args)
        self.item_encoder_logvar = Encoder(args)
        self.item_decoder = Decoder(args)
        self.args = args
        self.latent_dropout = nn.Dropout(args.reparam_dropout_rate)
        self.apply(self.init_weights)
        self.temperature = nn.Parameter(torch.zeros(1), requires_grad=True)

    def zero_attention_mask(self, input_ids):
        """
        :param input_ids: b*max_Sq*emb
        :return: b*1*max_se1_seq
        """
        return torch.zeros((1, 1, input_ids.shape[0], input_ids.shape[0]))

    def eps_anneal_function(self, step):

        return min(1.0, (1.0 * step) / self.args.total_annealing_step)

    def reparameterization(self, mu, logvar, step):  # vanila reparam

        std = torch.exp(0.5 * logvar)
        if self.training:
            eps = torch.randn_like(std)
            res = mu + std * eps
        else:
            res = mu + std
        return res

    def reparameterization1(self, mu, logvar):  # reparam without noise
        std = torch.exp(0.5 * logvar)
        return mu + std

    def reparameterization2(self, mu, logvar):  # use dropout
        if self.training:
            std = self.latent_dropout(torch.exp(0.5 * logvar))
        else:
            std = torch.exp(0.5 * logvar)
        res = mu + std
        return res

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def encode(self, sequence_emb, extended_attention_mask):  # forward

        item_encoded_mu_layers = self.item_encoder_mu(sequence_emb,
                                                      extended_attention_mask,
                                                      output_all_encoded_layers=True)

        item_encoded_logvar_layers = self.item_encoder_logvar(sequence_emb, extended_attention_mask,
                                                              True)

        return item_encoded_mu_layers[-1], item_encoded_logvar_layers[-1]

    def decode(self, z, extended_attention_mask):
        item_decoder_layers = self.item_decoder(z,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True)
        sequence_output = item_decoder_layers[-1]
        return sequence_output

    def forward(self, input_ids):
        zero_attention_mask = self.zero_attention_mask(input_ids).to(self.args.device)

        mu1, log_var1 = self.encode(input_ids, zero_attention_mask)
        mu2, log_var2 = self.encode(input_ids, zero_attention_mask)

        z1 = self.reparameterization1(mu1, log_var1)  # TODO: test
        z2 = self.reparameterization2(mu2, log_var2)

        reconstructed_seq1 = self.decode(z1, zero_attention_mask)
        reconstructed_seq2 = self.decode(z2, zero_attention_mask)

        return reconstructed_seq1, reconstructed_seq2, mu1, mu2, log_var1, log_var2, z1, z2


if __name__ == '__main__':
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

    args = parser.parse_args()
    model = ContrastVAE(args=args)
    relation = torch.ones(1, 1, 200)  # b * max_seq * emb

    reconstructed_seq1, reconstructed_seq2, mu1, mu2, log_var1, log_var2, z1, z2 = model(relation)
