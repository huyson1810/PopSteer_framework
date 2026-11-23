"""LightGCN + item-side SAE (PopSteer variant I-R on LightGCN).

As SASRec_ISAE: a TopK SAE (the SAE class of LightGCN_SAE, side="item") is trained on the frozen, PROPAGATED item
embeddings of LightGCN (the final layer-mean table e_i used for scoring). At scoring time that table is replaced by
its reconstruction, optionally steered in SAE space:
    e'_i = W_dec TopAct_K(ReLU(z(e_i) + delta)) + b_pre,     s'(u, i) = e_u^T e'_i
User embeddings stay the original propagated LightGCN embeddings. Neuron labels and statistics are read from
./dataset/<name>/item/ (written by scripts/steer_eval.py --variant I-R from real head vs tail items).
"""
import torch

from recbole.model.general_recommender.lightgcn import LightGCN
from recbole.model.general_recommender.lightgcn_sae import SAE
from recbole.utils import InputType


class LightGCN_ISAE(LightGCN):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        checkpoint = torch.load(config["base_path"], map_location=self.device, weights_only=False)
        self.load_state_dict(checkpoint["state_dict"], strict=False)
        self.sae_module_i = SAE(config, side="item")
        for p in self.parameters():
            p.requires_grad = False
        for p in self.sae_module_i.parameters():
            p.requires_grad = True
        self.base_u, self.base_i = None, None            # frozen propagated embeddings (computed once)
        self.restore_user_e, self.restore_item_e = None, None   # scoring cache, reset by steer_eval per setting
        self.val_fvu_i = torch.tensor(0.0, device=self.device)
        self.val_fvu_u = torch.tensor(0.0, device=self.device)
        self._fvu_i = torch.tensor(0.0, device=self.device)

    def _base(self):
        if self.base_i is None:
            with torch.no_grad():
                self.base_u, self.base_i = LightGCN.forward(self)
        return self.base_u, self.base_i

    def _item_table(self, train_mode=False):
        """Reconstructed (and, if steering is on, steered) scoring table; row 0 is the padding item."""
        _, base_i = self._base()
        rec = self.sae_module_i(base_i[1:], train_mode=train_mode)
        return torch.cat([base_i[:1], rec], dim=0)

    def calculate_loss(self, interaction):
        # the batch is ignored: every step fits the SAE to the full (frozen) item table, as SASRec_ISAE
        self.restore_user_e, self.restore_item_e = None, None
        self._item_table(train_mode=True)
        return self.sae_module_i.fvu + self.sae_module_i.auxk_loss / 2

    def _scoring(self):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e = self._base()[0]
            self.restore_item_e = self._item_table()
            self._fvu_i = self.sae_module_i.fvu.detach()
        return self.restore_user_e, self.restore_item_e

    def predict(self, interaction):
        u, i = self._scoring()
        return torch.mul(u[interaction[self.USER_ID]], i[interaction[self.ITEM_ID]]).sum(dim=1)

    def full_sort_predict(self, interaction):
        u, i = self._scoring()
        scores = torch.matmul(u[interaction[self.USER_ID]], i.transpose(0, 1))
        scores[:, 0] = float("-inf")
        self.val_fvu_i += self._fvu_i
        return scores.view(-1)
