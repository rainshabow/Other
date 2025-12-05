import torch
import torch.nn as nn

class RBM(nn.Module):
    def __init__(self, n_vis, n_hid, k=1):
        super().__init__()
        self.n_vis = n_vis
        self.n_hid = n_hid
        self.k = k
        self.W = nn.Parameter(torch.randn(n_hid, n_vis) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(n_vis))
        self.h_bias = nn.Parameter(torch.zeros(n_hid))

    def sample_h(self, v):
        prob = torch.sigmoid(torch.matmul(v, self.W.t()) + self.h_bias)
        return prob, torch.bernoulli(prob)

    def sample_v(self, h):
        prob = torch.sigmoid(torch.matmul(h, self.W) + self.v_bias)
        return prob, torch.bernoulli(prob)

    def forward(self, v):
        prob_h, h = self.sample_h(v)
        return prob_h, h

    def contrastive_divergence(self, v0, lr=1e-3):
        v = v0
        prob_h0, h0 = self.sample_h(v)
        for _ in range(self.k):
            prob_v, v = self.sample_v(h0)
            prob_h, h = self.sample_h(v)

        # 更新规则（对比散度 CD-k）
        positive = torch.matmul(prob_h0.t(), v0)
        negative = torch.matmul(prob_h.t(), v)

        batch_size = v0.size(0)
        dW = (positive - negative) / batch_size
        dvb = torch.mean(v0 - v, dim=0)
        dhb = torch.mean(prob_h0 - prob_h, dim=0)

        self.W.data += lr * dW
        self.v_bias.data += lr * dvb
        self.h_bias.data += lr * dhb

        # 重构误差（用于监控训练质量）
        loss = torch.mean((v0 - v) ** 2)
        return loss.item()
