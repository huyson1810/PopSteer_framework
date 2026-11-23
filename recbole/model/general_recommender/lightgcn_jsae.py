"""LightGCN + ONE SAE shared by user and item embeddings (PopSteer variant UI-R on LightGCN).

As SASRec_JSAE: each modality m in {user, item} is standardised with its own mean and scalar scale before the shared
SAE and de-standardised after decoding,  x_hat = sd_m * SAE((x - mu_m) / sd_m) + mu_m.  Both statistics come from the
frozen PROPAGATED LightGCN embeddings (computed once, then stored as buffers in the checkpoint).
Training: every step fits the SAE to the batch's users AND an equally large random sample of items (50/50 balance;
loss = PopSteer's fvu + auxk/2 for each modality).
Steering: `steer_side` in {"user", "item", "both"}; `item_gate` applies the "gate" rule on the item side only. No
synthetic users exist for LightGCN, so only real-user labels (UI-R) are used. The SAE is exposed as
`sae_module_u` so the real-user labels of scripts/steer_eval.py work unchanged (after standardising, see `std_u`).
"""
import torch

from recbole.model.general_recommender.lightgcn import LightGCN
from recbole.model.general_recommender.lightgcn_sae import SAE
from recbole.utils import InputType


class LightGCN_JSAE(LightGCN):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        checkpoint = torch.load(config["base_path"], map_location=self.device, weights_only=False)
        self.load_state_dict(checkpoint["state_dict"], strict=False)
        self.sae_module_u = SAE(config, side="user")      # the shared dictionary
        for p in self.parameters():
            p.requires_grad = False
        for p in self.sae_module_u.parameters():
            p.requires_grad = True
        d = config["embedding_size"]
        for m in ("u", "i"):
            self.register_buffer(f"mu_{m}", torch.zeros(d))
            self.register_buffer(f"sd_{m}", torch.ones(1))
        self.register_buffer("stats_ready", torch.zeros(1))
        self.steer_side = "both"
        self.item_gate = False
        self.item_align = False
        self.base_u, self.base_i = None, None
        self.restore_user_e, self.restore_item_e = None, None
        self.val_fvu_u = torch.tensor(0.0, device=self.device)
        self.val_fvu_i = torch.tensor(0.0, device=self.device)
        self._fvu = (torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device))

    def _base(self):
        if self.base_u is None:
            with torch.no_grad():
                self.base_u, self.base_i = LightGCN.forward(self)
            if self.stats_ready.item() == 0:      # first call of a fresh model; a loaded checkpoint has them
                for m, x in (("u", self.base_u[1:]), ("i", self.base_i[1:])):
                    mu = x.mean(0)
                    getattr(self, f"mu_{m}").copy_(mu)
                    getattr(self, f"sd_{m}").copy_(((x - mu).pow(2).sum(1).mean() / x.shape[1]).sqrt().reshape(1))
                self.stats_ready.fill_(1)
        return self.base_u, self.base_i

    def std_u(self, x):
        return (x - self.mu_u) / self.sd_u

    def _side(self, x, m, steer=False, gate=False, align=False, train_mode=False):
        sae = self.sae_module_u
        mu, sd = getattr(self, f"mu_{m}"), getattr(self, f"sd_{m}")
        sae.steer, sae.gate, sae.use_align = steer, gate, align
        r = sae((x - mu) / sd, train_mode=train_mode) * sd + mu
        sae.gate, sae.use_align = False, False
        return r

    def calculate_loss(self, interaction):
        self.restore_user_e, self.restore_item_e = None, None
        base_u, base_i = self._base()
        users = interaction[self.USER_ID]
        sae = self.sae_module_u
        master = sae.steer
        self._side(base_u[users], "u", train_mode=True)
        loss_u = sae.fvu + sae.auxk_loss / 2
        idx = torch.randint(1, base_i.shape[0], (users.shape[0],), device=users.device)
        self._side(base_i[idx], "i", train_mode=True)
        loss_i = sae.fvu + sae.auxk_loss / 2
        sae.steer = master
        return loss_u + loss_i

    def _scoring(self):
        if self.restore_user_e is None or self.restore_item_e is None:
            base_u, base_i = self._base()
            sae = self.sae_module_u
            master = sae.steer
            u = self._side(base_u[1:], "u", steer=master and self.steer_side in ("user", "both"))
            fvu_u = sae.fvu.detach()
            i = self._side(base_i[1:], "i", steer=master and self.steer_side in ("item", "both"),
                           gate=self.item_gate, align=self.item_align)
            fvu_i = sae.fvu.detach()
            sae.steer = master
            self.restore_user_e = torch.cat([base_u[:1], u], dim=0)
            self.restore_item_e = torch.cat([base_i[:1], i], dim=0)
            self._fvu = (fvu_u, fvu_i)
        return self.restore_user_e, self.restore_item_e

    def predict(self, interaction):
        u, i = self._scoring()
        return torch.mul(u[interaction[self.USER_ID]], i[interaction[self.ITEM_ID]]).sum(dim=1)

    def full_sort_predict(self, interaction):
        u, i = self._scoring()
        scores = torch.matmul(u[interaction[self.USER_ID]], i.transpose(0, 1))
        scores[:, 0] = float("-inf")
        self.val_fvu_u += self._fvu[0]
        self.val_fvu_i += self._fvu[1]
        return scores.view(-1)
