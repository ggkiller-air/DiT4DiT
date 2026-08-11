import numpy as np
import pytest

from deployment.model_server.sonic_policy import (
    SonicPolicyAdapter,
    SonicQ99Normalizer,
    make_stereo_condition,
    validate_actions,
    validate_observation,
)


def observation():
    return {
        "state": np.zeros(46, dtype=np.float32),
        "ego_view_left": np.zeros((8, 10, 3), dtype=np.uint8),
        "ego_view_right": np.zeros((8, 10, 3), dtype=np.uint8),
        "prompt": "carry the bucket",
        "tactile": np.zeros(768, dtype=np.uint8),
    }


def test_sonic_observation_and_stereo_contract():
    obs = validate_observation(observation(), requires_tactile=True)
    packed = make_stereo_condition(obs["ego_view_left"], obs["ego_view_right"])
    assert tuple(packed.shape) == (3, 224, 448)


def test_sonic_observation_requires_tactile_for_jepa_checkpoint():
    obs = observation()
    del obs["tactile"]
    with pytest.raises(ValueError, match="requires tactile"):
        validate_observation(obs, requires_tactile=True)


def test_sonic_q99_normalization_matches_training_formula_without_binary_channel():
    state_stats = {"q01": [-2.0] * 46, "q99": [2.0] * 46}
    action_stats = {"q01": [-4.0] * 78, "q99": [4.0] * 78}
    normalizer = SonicQ99Normalizer(state_stats, action_stats)

    np.testing.assert_allclose(
        normalizer.normalize_state(np.ones(46, dtype=np.float32)),
        np.full(46, 0.5, dtype=np.float32),
    )
    normalized = np.zeros((40, 78), dtype=np.float32)
    normalized[:, 6] = 0.25
    result = normalizer.unnormalize_actions(normalized)
    np.testing.assert_allclose(result[:, 6], 1.0)


def test_sonic_action_contract_rejects_wrong_shape_and_nonfinite():
    with pytest.raises(ValueError, match="shape"):
        validate_actions(np.zeros((16, 32), dtype=np.float32))
    actions = np.zeros((40, 78), dtype=np.float32)
    actions[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinity"):
        validate_actions(actions)


class FakeDiTPolicy:
    def __init__(self):
        self.example = None

    @staticmethod
    def _check_unnorm_key(_norm_stats, unnorm_key):
        return unnorm_key or "desk_sweep"

    def predict_action(self, *, examples):
        self.example = examples[0]
        return {"normalized_actions": np.zeros((1, 40, 78), dtype=np.float32)}


def test_sonic_adapter_matches_dit4dit_training_inputs_and_output_stats():
    policy = FakeDiTPolicy()
    config = {
        "framework": {
            "action_model": {
                "action_dim": 78,
                "action_horizon": 40,
                "state_dim": 46,
                "tactile_mode": "dream",
            }
        }
    }
    norm_stats = {
        "desk_sweep": {
            "state": {"q01": [-2.0] * 46, "q99": [2.0] * 46},
            "action": {"q01": [-4.0] * 78, "q99": [4.0] * 78},
        }
    }
    adapter = SonicPolicyAdapter(
        policy,
        model_config=config,
        norm_stats=norm_stats,
        unnorm_key="desk_sweep",
    )
    obs = observation()
    obs["state"].fill(1.0)

    result = adapter.infer(obs)

    assert result["actions"].shape == (40, 78)
    assert policy.example is not None
    assert len(policy.example["image"]) == 1
    assert tuple(policy.example["image"][0].shape) == (3, 224, 448)
    np.testing.assert_allclose(policy.example["state"], np.full(46, 0.5))
    np.testing.assert_array_equal(policy.example["tactile"], obs["tactile"])
    np.testing.assert_allclose(result["actions"], 0.0)
