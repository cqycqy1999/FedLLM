from __future__ import annotations

import copy


class ReferenceModelManager:
    def __init__(self, cfg):
        self.cfg = cfg

    def build(self, policy_model):
        ref_model = copy.deepcopy(policy_model)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        return ref_model