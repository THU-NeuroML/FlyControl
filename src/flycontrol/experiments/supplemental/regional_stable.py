from . import regional_evaluate as original
from contextlib import contextmanager
import numpy as np,torch
original_weights=original.regional_weights
class StableReplay(original.RegionalReplay):
    @contextmanager
    def messages(self):
        if self.mode not in self.weights:
            yield
            return
        core = self.actor.core
        ids, fraction = self.weights[self.mode]
        self.message_count = len(ids)
        self.fractional_abs_weight_sum = float((core.edge_weight[ids] * fraction).abs().sum())
        ep, eo, ew = core.edge_pre, core.edge_post, core.edge_weight
        assert '_recurrent_drive' not in core.__dict__
        def replay_drive(r):
            ref = torch.tanh(torch.from_numpy(self.baseline['hidden_before'][self.index]))
            source = r[:, ep]
            selected = source[:, ids]
            reference = ref[ep[ids]][None]
            # f=1 uses the reference exactly; f=0 was excluded; r=reference
            # preserves native multiplication/summation bit for bit.
            replacement = torch.where(fraction[None] == 1, reference,
                selected + (reference - selected) * fraction[None])
            blended = source.clone()
            blended[:, ids] = replacement
            messages = blended * ew
            correction = torch.zeros_like(r).index_add_(1, eo[ids], (replacement - selected) * ew[ids])
            self.correction_l1.append(float(correction.abs().sum()))
            return torch.zeros_like(r).index_add_(1, eo, messages)
        core._recurrent_drive = replay_drive
        try:
            yield
        finally:
            del core._recurrent_drive
def stable_weights(core, fractions, masks):
    result = original_weights(core, fractions, masks)
    # Brain and VNC are the complement of neck in the frozen source partition.
    # This makes the no-neck case exactly one, not 1 +/- rounding of b + v.
    f = 1. - fractions['neck_fraction']
    ids = np.flatnonzero(f)
    result['brain_vnc_reference'] = (torch.from_numpy(ids), torch.from_numpy(f[ids].copy()))
    return result
