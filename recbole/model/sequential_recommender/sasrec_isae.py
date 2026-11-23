"""SASRec + item-side SAE (PopSteer variant I-R).

A TopK SAE (the same SAE class as PopSteer, side="item") is trained on the frozen SASRec item table. At scoring
time the table is replaced by its reconstruction, optionally steered in SAE space:
    q'_i = W_dec TopAct_K(ReLU(z(q_i) + delta)) + b_pre,     s'(u, i) = p_u^T q'_i
Only the SCORING table is replaced; the item embeddings that encode the user's history stay unchanged, so p_u is
the original SASRec user state. Neuron labels and statistics are read from ./dataset/<name>/item/ (written by
scripts/steer_eval.py --variant I-R from real head vs tail items), exactly as the user SAE reads ./dataset/<name>/user/.
"""
import torch

from recbole.model.sequential_recommender.sasrec import SASRec
from recbole.model.sequential_recommender.sasrec_sae import SAE


class SASRec_ISAE(SASRec):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        checkpoint = torch.load(config["base_path"], map_location=self.device, weights_only=False)
        self.load_state_dict(checkpoint["state_dict"], strict=False)
        self.sae_module_i = SAE(config, side="item")
        for p in self.parameters():
            p.requires_grad = False
        for p in self.sae_module_i.parameters():
            p.requires_grad = True
        # validation reconstruction loss (metric SAE_Loss_i, collected and reset by the evaluator)
        self.val_fvu_i = torch.tensor(0.0, device=self.device)
        self.val_fvu_u = torch.tensor(0.0, device=self.device)

    def _item_table(self, train_mode=False):
        """Reconstructed (and, if steering is on, steered) scoring table; row 0 is the padding item."""
        table = self.item_embedding.weight
        self.sae_module_i.noise_fixed_rows = True     # noise ablation: one fixed draw per item (sasrec_sae.SAE)
        rec = self.sae_module_i(table[1:], train_mode=train_mode)
        return torch.cat([table[:1], rec], dim=0)

    def calculate_loss(self, interaction):
        # the batch is ignored: every step fits the SAE to the full (frozen) item table, as PopSteer's loss
        self.sae_module_i(self.item_embedding.weight[1:].detach(), train_mode=True)
        return self.sae_module_i.fvu + self.sae_module_i.auxk_loss / 2

    def predict(self, interaction):
        seq_output = SASRec.forward(self, interaction[self.ITEM_SEQ], interaction[self.ITEM_SEQ_LEN])
        table = self._item_table()
        return torch.mul(seq_output, table[interaction[self.ITEM_ID]]).sum(dim=1)

    def full_sort_predict(self, interaction):
        seq_output = SASRec.forward(self, interaction[self.ITEM_SEQ], interaction[self.ITEM_SEQ_LEN])
        table = self._item_table()
        self.val_fvu_i += self.sae_module_i.fvu.detach()
        return torch.matmul(seq_output, table.transpose(0, 1))
