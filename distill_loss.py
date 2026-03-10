import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillLoss(nn.Module):
    """
    Combined loss for student training.

    Components:
        - KD loss: KL(student_probs || teacher_probs) with temperature
        - Regression loss: L1(student_score, teacher_score)
        - Optional CE loss for human-labeled samples
        - Optional ranking loss within batch
    """

    def __init__(
        self,
        temperature: float = 1.0,
        lambda_kd: float = 1.0,
        lambda_reg: float = 1.0,
        lambda_ce: float = 1.0,
        lambda_rank: float = 0.0,
        rank_margin: float = 0.05,
        rank_delta: float = 0.10,
        use_conf_weight: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.lambda_kd = lambda_kd
        self.lambda_reg = lambda_reg
        self.lambda_ce = lambda_ce
        self.lambda_rank = lambda_rank
        self.rank_margin = rank_margin
        self.rank_delta = rank_delta
        self.use_conf_weight = use_conf_weight

        self.ce = nn.CrossEntropyLoss(reduction="none")

    def kd_loss(self, student_logits, teacher_probs):
        """
        KL divergence between teacher_probs and softened student probs.
        teacher_probs is assumed already normalized.
        """
        T = self.temperature
        log_p_s = F.log_softmax(student_logits / T, dim=1)
        p_t = teacher_probs.clamp(min=1e-8)
        kd = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=1) * (T * T)
        return kd

    def reg_loss(self, student_score, teacher_score):
        """
        student_score: [B, 1]
        teacher_score: [B]
        """
        teacher_score = teacher_score.view(-1, 1)
        reg = F.l1_loss(student_score, teacher_score, reduction="none").squeeze(1)
        return reg

    def ce_loss(self, student_logits, gt_class):
        return self.ce(student_logits, gt_class)

    def rank_loss(self, student_score, teacher_score):
        """
        Pairwise ranking loss within batch.

        teacher_score: [B]
        student_score: [B, 1]
        """
        s = student_score.view(-1)
        t = teacher_score.view(-1)

        diff_t = t.unsqueeze(1) - t.unsqueeze(0)
        diff_s = s.unsqueeze(1) - s.unsqueeze(0)

        valid = diff_t > self.rank_delta
        if valid.sum() == 0:
            return s.new_tensor(0.0)

        loss_mat = F.relu(self.rank_margin - diff_s)
        loss = loss_mat[valid].mean()
        return loss

    def forward(self, outputs, batch):
        """
        outputs:
            logits, score
        batch:
            teacher_probs: [B, C]
            teacher_score: [B]
            teacher_conf:  [B]
            gt_class:      [B]
            is_human_labeled: [B]
        """
        student_logits = outputs["logits"]
        student_score = outputs["score"]

        teacher_probs = batch["teacher_probs"]
        teacher_score = batch["teacher_score"]
        teacher_conf = batch["teacher_conf"]
        gt_class = batch["gt_class"]
        is_human_labeled = batch["is_human_labeled"].float()

        kd = self.kd_loss(student_logits, teacher_probs)
        reg = self.reg_loss(student_score, teacher_score)

        if self.use_conf_weight:
            weight = teacher_conf
            kd = kd * weight
            reg = reg * weight

        kd = kd.mean()
        reg = reg.mean()

        ce = student_logits.new_tensor(0.0)
        if is_human_labeled.sum() > 0:
            mask = is_human_labeled > 0.5
            ce_vals = self.ce_loss(student_logits[mask], gt_class[mask])
            ce = ce_vals.mean()

        rank = student_logits.new_tensor(0.0)
        if self.lambda_rank > 0:
            rank = self.rank_loss(student_score, teacher_score)

        total = (
            self.lambda_kd * kd
            + self.lambda_reg * reg
            + self.lambda_ce * ce
            + self.lambda_rank * rank
        )

        loss_dict = {
            "loss": total,
            "loss_kd": kd.detach(),
            "loss_reg": reg.detach(),
            "loss_ce": ce.detach(),
            "loss_rank": rank.detach(),
        }
        return loss_dict
