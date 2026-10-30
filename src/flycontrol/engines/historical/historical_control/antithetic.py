"""Antithetic innovations across environment halves; Gaussian marginals unchanged."""
import hashlib
import torch


def install_antithetic(policy):
    assert not hasattr(policy, '_antithetic_audit')
    native_act = policy.act
    policy._antithetic_audit = None

    @torch.no_grad()
    def act(obs, hidden, deterministic=False, generator=None):
        if deterministic:
            return native_act(obs, hidden, deterministic=True, generator=generator)
        assert generator is not None, 'Training uses the explicit CPU sampling stream.'
        mean, state, value = policy.forward(obs, hidden)
        assert mean.ndim == 2 and mean.shape[0] >= 2 and mean.shape[0] % 2 == 0
        noise = torch.randn(mean.shape, dtype=mean.dtype, device='cpu', generator=generator)
        raw_noise = noise.clone()
        half = len(noise) // 2
        noise[half:] = -noise[:half]
        assert torch.equal(noise[:half], raw_noise[:half])
        assert torch.count_nonzero(noise[:half] + noise[half:]).item() == 0
        audit = policy._antithetic_audit
        assert audit is not None, 'Reset the sampling audit before collecting a rollout.'
        audit['shapes'].append(tuple(noise.shape))
        audit['raw'].update(raw_noise.numpy().tobytes())
        audit['used'].update(noise.numpy().tobytes())
        distribution = policy.distribution(mean)
        raw = (mean.cpu() + policy.log_std.cpu().exp() * noise).to(mean.device)
        return raw, distribution.log_prob(raw).sum(-1), value, state

    policy.act = act
    return policy


def reset_audit(policy, generator):
    policy._antithetic_audit = dict(initial_state=generator.get_state().clone(), shapes=[],
                                   raw=hashlib.sha256(), used=hashlib.sha256())


def finish_audit(policy, generator, expected_calls, expected_batch):
    audit = policy._antithetic_audit
    assert len(audit['shapes']) == expected_calls
    assert all(shape == (expected_batch, 6) for shape in audit['shapes'])
    reference = torch.Generator('cpu')
    reference.set_state(audit['initial_state'])
    raw_hash, used_hash = hashlib.sha256(), hashlib.sha256()
    for shape in audit['shapes']:
        sample = torch.randn(shape, dtype=torch.float32, device='cpu', generator=reference)
        raw_hash.update(sample.numpy().tobytes())
        half = shape[0] // 2
        paired = torch.cat((sample[:half], -sample[:half]), dim=0)
        used_hash.update(paired.numpy().tobytes())
    assert raw_hash.hexdigest() == audit['raw'].hexdigest()
    assert used_hash.hexdigest() == audit['used'].hexdigest()
    assert torch.equal(generator.get_state(), reference.get_state())
    return dict(passed=True, calls=expected_calls, environments=expected_batch,
        raw_full_batch_innovations_sha256=raw_hash.hexdigest(), used_antithetic_innovations_sha256=used_hash.hexdigest(),
        initial_generator_sha256=hashlib.sha256(audit['initial_state'].numpy().tobytes()).hexdigest(),
        final_generator_sha256=hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest(),
        original_full_batch_draw_count_preserved=True, first_half_preserved=True, second_half_exact_negation=True,
        interpretation='Every environment retains a standard Gaussian marginal; simultaneous environments are deliberately correlated. No claim that return or gradient variance must decrease.')
