"""SASRec + ONE SAE shared by user states and item embeddings (PopSteer variants UI-S and UI-R).

Each modality m in {user, item} is standardised with its own mean and scalar scale before the shared SAE and
de-standardised after decoding:  x_hat = sd_m * SAE((x - mu_m) / sd_m) + mu_m.
  - item statistics: from the frozen scoring table (fixed);
  - user statistics: from the first training batch of SASRec user states (then frozen).
Training: every step fits the SAE to the batch's user states AND an equally large random sample of item rows
(50/50 balance; loss = PopSteer's fvu + auxk/2 for each modality). Only the scoring table is replaced on the item
side; the embeddings that encode histories are unchanged.
Steering: `steer_side` in {"user", "item", "both"} chooses where the (single) steering vector is applied; with the
reconstructed score s' = p'_u^T q'_i. The SAE is exposed as `sae_module_u`, so PopSteer's synthetic-user analysis
(trainer.analyze_neurons) and the real-user labels of scripts/steer_eval.py work unchanged on the user side.
"""
import torch

from recbole.model.sequential_recommender.sasrec import SASRec
from recbole.model.sequential_recommender.sasrec_sae import SAE
from recbole.utils import save_batch_activations, make_items_popular, make_items_unpopular


class SASRec_JSAE(SASRec):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        checkpoint = torch.load(config["base_path"], map_location=self.device, weights_only=False)
        self.load_state_dict(checkpoint["state_dict"], strict=False)
        self.sae_module_u = SAE(config, side="user")      # the shared dictionary
        for p in self.parameters():
            p.requires_grad = False
        for p in self.sae_module_u.parameters():
            p.requires_grad = True
        table = self.item_embedding.weight.detach()[1:]
        mu_i = table.mean(0)
        self.register_buffer("mu_i", mu_i)
        self.register_buffer("sd_i", ((table - mu_i).pow(2).sum(1).mean() / table.shape[1]).sqrt().reshape(1))
        self.register_buffer("mu_u", torch.zeros_like(mu_i))
        self.register_buffer("sd_u", torch.ones(1, device=mu_i.device))
        self.register_buffer("user_stats_ready", torch.zeros(1, device=mu_i.device))
        self.steer_side = "both"
        self.item_gate = False      # "gate" rule on the item side (steer_eval.py --gate)
        self.item_align = False     # "align" rule on the item side (steer_eval.py --align)
        # validation reconstruction losses (metrics SAE_Loss_u / _i / _total, collected and reset by the evaluator)
        self.val_fvu_u = torch.tensor(0.0, device=self.device)
        self.val_fvu_i = torch.tensor(0.0, device=self.device)

    # -------------------------------------------------------------- encoders
    def _user_rec(self, item_seq, item_seq_len, steer=False, train_mode=False):
        x = SASRec.forward(self, item_seq, item_seq_len)
        self.sae_module_u.steer = steer
        r = self.sae_module_u((x - self.mu_u) / self.sd_u, train_mode=train_mode)
        return r * self.sd_u + self.mu_u

    def _item_table(self, steer=False):
        table = self.item_embedding.weight
        self.sae_module_u.steer = steer
        self.sae_module_u.gate, self.sae_module_u.use_align = self.item_gate, self.item_align
        self.sae_module_u.noise_fixed_rows = True     # noise ablation: one fixed draw per item (sasrec_sae.SAE)
        r = self.sae_module_u((table[1:] - self.mu_i) / self.sd_i) * self.sd_i + self.mu_i
        self.sae_module_u.gate, self.sae_module_u.use_align = False, False
        self.sae_module_u.noise_fixed_rows = False
        return torch.cat([table[:1], r], dim=0)

    # -------------------------------------------------------------- training
    def calculate_loss(self, interaction):
        x = SASRec.forward(self, interaction[self.ITEM_SEQ], interaction[self.ITEM_SEQ_LEN]).detach()
        if self.user_stats_ready.item() == 0:
            mu_u = x.mean(0)
            self.mu_u.copy_(mu_u)
            self.sd_u.copy_(((x - mu_u).pow(2).sum(1).mean() / x.shape[1]).sqrt().reshape(1))
            self.user_stats_ready.fill_(1)
        sae = self.sae_module_u
        sae.steer = False
        sae((x - self.mu_u) / self.sd_u, train_mode=True)
        loss_u = sae.fvu + sae.auxk_loss / 2
        idx = torch.randint(1, self.item_embedding.weight.shape[0], (x.shape[0],), device=x.device)
        q = (self.item_embedding.weight[idx].detach() - self.mu_i) / self.sd_i
        sae(q, train_mode=True)
        loss_i = sae.fvu + sae.auxk_loss / 2
        return loss_u + loss_i

    # -------------------------------------------------------------- scoring
    def predict(self, interaction):
        master = self.sae_module_u.steer
        xu = self._user_rec(interaction[self.ITEM_SEQ], interaction[self.ITEM_SEQ_LEN],
                            steer=master and self.steer_side in ("user", "both"))
        table = self._item_table(steer=master and self.steer_side in ("item", "both"))
        self.sae_module_u.steer = master
        return torch.mul(xu, table[interaction[self.ITEM_ID]]).sum(dim=1)

    def full_sort_predict(self, interaction, popular=None, save=False):
        item_seq, item_seq_len = interaction[self.ITEM_SEQ], interaction[self.ITEM_SEQ_LEN]
        master = self.sae_module_u.steer
        if popular is not None:     # PopSteer's synthetic-user analysis (trainer.analyze_neurons)
            maker = make_items_popular if popular else make_items_unpopular
            seq = maker(item_seq.shape[0], self.dataset, self.max_seq_length).to(self.device)
            self._user_rec(seq, item_seq_len, steer=False)
            save_batch_activations(self.sae_module_u.last_activations, self.sae_module_u.hidden_dim,
                                   self.dataset, popular)
            self.sae_module_u.steer = master
            return
        xu = self._user_rec(item_seq, item_seq_len, steer=master and self.steer_side in ("user", "both"))
        self.val_fvu_u += self.sae_module_u.fvu.detach()
        if save:
            save_batch_activations(self.sae_module_u.last_activations, self.sae_module_u.hidden_dim,
                                   self.dataset, popular, steered=False)
        table = self._item_table(steer=master and self.steer_side in ("item", "both"))
        self.val_fvu_i += self.sae_module_u.fvu.detach()
        self.sae_module_u.steer = master
        return torch.matmul(xu, table.transpose(0, 1))
