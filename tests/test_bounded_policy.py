import unittest
import torch
from tensordict import TensorDict
from torch.distributions import Normal, TransformedDistribution, TanhTransform, AffineTransform
from spot_micro.bounded_policy import SquashedActorCritic
from spot_micro.export import ExportedActor


class BoundedPolicyTests(unittest.TestCase):
    def make(self, n=64):
        obs = TensorDict({"policy": torch.randn(n,4), "critic": torch.randn(n,6)}, [n])
        policy = SquashedActorCritic(obs, {"policy":["policy"], "critic":["critic"]}, 2,
                                    actor_hidden_dims=[8], critic_hidden_dims=[8], noise_std_type="log")
        return policy, obs

    def test_density_matches_pytorch_change_of_variables(self):
        torch.manual_seed(12)
        p, obs = self.make()
        action = p.act(obs)
        reference = TransformedDistribution(p.distribution, [TanhTransform(), AffineTransform(0, p.action_postprocessor.action_bound)])
        torch.testing.assert_close(p.get_actions_log_prob(action), reference.log_prob(action).sum(-1), atol=2e-5, rtol=2e-5)

    def test_support_and_noise_limits(self):
        p, obs = self.make(10000)
        for raw in [-1000., 1000.]:
            with torch.no_grad():
                p.noise_logit.fill_(raw)
            self.assertTrue((p.latent_std >= .05).all())
            self.assertTrue((p.latent_std <= .5).all())
            action = p.act(obs)
            self.assertTrue((action.abs() < 1).all())
            self.assertTrue(torch.isfinite(p.get_actions_log_prob(action)).all())

    def test_mapping_preserves_existing_clipped_actions(self):
        from spot_micro.bounded_policy import ActionSquash
        transform = ActionSquash()
        grid = torch.linspace(-100,100,100001)
        self.assertLess(float((transform(grid)-grid.clamp(-1,1)).abs().max()),.0026)
        interior = torch.linspace(-.98,.98,1000)
        torch.testing.assert_close(transform(interior),interior,atol=1.1e-5,rtol=0)

    def test_density_rejects_values_outside_support(self):
        p, obs = self.make()
        p.act(obs)
        self.assertTrue(torch.isneginf(p.get_actions_log_prob(torch.ones(64,2))).all())

    def test_large_old_actor_logits_keep_finite_training_quantities(self):
        p, obs = self.make(10000)
        with torch.no_grad():
            p.actor[-1].weight.zero_()
            p.actor[-1].bias.copy_(torch.tensor([-1000.,1000.]))
            p.noise_logit.fill_(30.)
        actions = p.act(obs)
        loss = p.get_actions_log_prob(actions).mean()+p.entropy.mean()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(torch.isfinite(v.grad).all() for v in p.parameters() if v.grad is not None))

    def test_entropy_gradient_is_finite_and_includes_squash(self):
        torch.manual_seed(20)
        p, obs = self.make(4096)
        with torch.no_grad():
            p.actor[-1].weight.zero_()
            p.actor[-1].bias.fill_(3.)
        p.act(obs)
        entropy = p.entropy.mean()
        self.assertLess(float(entropy), float(p.distribution.entropy().sum(-1).mean())-1)
        entropy.backward()
        self.assertTrue(torch.isfinite(p.noise_logit.grad).all())
        self.assertTrue((p.actor[-1].bias.grad < 0).all())

    def test_export_and_traced_inference_keep_transform(self):
        p, obs = self.make()
        expected = p.act_inference(obs)
        model = ExportedActor(p).eval()
        torch.testing.assert_close(model(obs["policy"]), expected)
        traced = torch.jit.trace(model, obs["policy"][:1])
        torch.testing.assert_close(traced(obs["policy"]), expected)

    def test_ppo_updates_use_bounded_actions_and_remain_finite(self):
        from rsl_rl.algorithms import PPO
        p, obs = self.make()
        alg = PPO(p, num_learning_epochs=2, num_mini_batches=2, device="cpu")
        alg.init_storage("rl",64,8,obs,[2])
        before = p.actor[-1].weight.detach().clone()
        for _ in range(3):
            with torch.inference_mode():
                for _ in range(8):
                    actions = alg.act(obs)
                    self.assertTrue((actions.abs() < 1).all())
                    reward = -(actions-.2).square().sum(-1)
                    alg.process_env_step(obs,reward,torch.zeros(64,dtype=torch.bool),{})
                alg.compute_returns(obs)
            losses = alg.update()
            self.assertTrue(all(torch.isfinite(torch.as_tensor(v)).all() for v in losses.values() if v is not None))
        self.assertFalse(torch.equal(before,p.actor[-1].weight))
        self.assertTrue(all(torch.isfinite(v).all() for v in p.parameters()))


if __name__ == "__main__":
    unittest.main()
