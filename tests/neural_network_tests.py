import unittest
import sys
import numpy as np
import torch
sys.path.append("..")
from games.guessit import TwoPlayerGuessIt
from neural_network import NeuralNetwork
from models.dumbnet import DumbNet
from models.mlp import MLP




class TrainTests(unittest.TestCase):

    def test_easy_train(self):
        gi = TwoPlayerGuessIt()
        nn = NeuralNetwork(gi, MLP, weight_decay=0, lr=1e-3)

        data = []
        s1 = gi.get_initial_state()
        s2 = gi.take_action(s1, np.array([[1,0],[0,0]]))
        s3 = gi.take_action(s2, np.array([[0,1],[0,0]]))
        data.append([s1, np.array([.1,.1,.1,.7]), .65])
        data.append([s2, np.array([.1,.1,.8]), .30])
        data.append([s3, np.array([.4,.6]), .1])
        data = np.array(data)

        for _ in range(350):
            nn.train(data)

        for row in data:
            s = row[0]
            p, v = nn.predict(s)
            np.testing.assert_allclose(p, row[1].astype(np.float32), atol=1e-5)
            np.testing.assert_allclose(v, row[2], atol=1e-5)

    def test_tie_train(self):
        gi = TwoPlayerGuessIt()
        nn = NeuralNetwork(gi, MLP, weight_decay=0, lr=1e-4)

        data = []
        s1 = gi.get_initial_state()
        data.append([s1, np.array([.3,.2,.1,.4]), .8])
        data.append([s1, np.array([.4,.3,.2,.1]), .0])
        # Should become [.35, .25, .15, .25], .4
        data = np.array(data)

        for _ in range(600):
            nn.train(data)
        p, v = nn.predict(s1)
        np.testing.assert_allclose(p, np.array([.35, .25, .15, .25], dtype=np.float32), atol=0.01)
        np.testing.assert_allclose(v, .4, atol=.03)



class PredictTests(unittest.TestCase):

    def test_dumbnet_predict(self):
        gi = TwoPlayerGuessIt()
        nn = NeuralNetwork(gi, DumbNet)
        init = gi.get_initial_state()
        p, v = nn.predict(init)
        self.assertEqual(len(p), 4)
        self.assertTrue((p == .25).all())
        self.assertEqual(v, 0.0)

        template = np.zeros_like(gi.get_available_actions(init))
        template[0, 1] = 1
        s = gi.take_action(init, template)
        p, v = nn.predict(s)
        self.assertEqual(len(p), 3)
        self.assertTrue((p == 0.3333333).all())
        self.assertEqual(v, 0.0)


class GetValidDistTests(unittest.TestCase):

    def test_uniform_get_valid_dist(self):
        gi = TwoPlayerGuessIt()
        nn = NeuralNetwork(gi, DumbNet)
        init = gi.get_initial_state()
        logits = torch.ones((2,2))
        self.assertEqual(list(nn.get_valid_dist(init, logits)), [.25, .25, .25, .25])
        self.assertEqual(list(nn.get_valid_dist(init, logits, log_softmax=True)), [-1.38629436]*4)
        
        template = np.zeros_like(gi.get_available_actions(init))
        template[0, 1] = 1
        s = gi.take_action(init, template)
        self.assertEqual(list(nn.get_valid_dist(s, logits)), [0.3333333]*3)
        self.assertEqual(list(nn.get_valid_dist(s, logits, log_softmax=True)), [-1.09861228]*3)

    def test_prior_get_valid_dist(self):
        gi = TwoPlayerGuessIt()
        nn = NeuralNetwork(gi, DumbNet)
        init = gi.get_initial_state()
        logits = torch.ones((2,2))
        logits[:,-1] = 2
        dist = [float(x) for x in list(nn.get_valid_dist(init, logits))]
        self.assertEqual(dist, [0.13447073101997375, 0.3655293583869934, 0.13447073101997375, 0.3655293583869934])

        template = np.zeros_like(gi.get_available_actions(init))
        template[0, 1] = 1
        s = gi.take_action(init, template)
        dist = [float(x) for x in list(nn.get_valid_dist(s, logits))]
        self.assertEqual(dist, [0.21194154024124146, 0.21194154024124146, 0.5761168599128723])
      

class LossTests(unittest.TestCase):


    def test_easy_loss(self):
        gi = TwoPlayerGuessIt()
        nn = NeuralNetwork(gi, DumbNet)
        states = np.array([gi.get_initial_state()])

        p_gt = np.ones(4) * .1
        p_gt[-1] = .7
        p_gt = np.array([p_gt])
        p_pred = np.ones((1,2,2), dtype=np.float32)
        p_pred[0,1,1] = 2
        p_pred = torch.from_numpy(p_pred)
        v_gt, v_pred = torch.Tensor([1.]), torch.Tensor([-.35])
        l = nn.loss(states, (p_pred, v_pred), (p_gt, v_gt))
        val = l.detach().numpy().reshape(-1)[0]

        mse_loss = 1.8225
        ce_loss = 1.0437
        self.assertAlmostEqual(mse_loss + ce_loss, val, places=4)


    def test_masked_loss(self):
        gi = TwoPlayerGuessIt()
        nn = NeuralNetwork(gi, DumbNet)
        s = gi.get_initial_state()
        template = np.zeros_like(gi.get_available_actions(s))
        template[0, 1] = 1
        s = gi.take_action(s, template)
        states = np.array([s])

        p_gt = np.ones(3) * .2
        p_gt[-1] = .6
        p_gt = np.array([p_gt])
        p_pred = np.ones((1,2,2), dtype=np.float32)
        p_pred[0,1,1] = 2
        p_pred = torch.from_numpy(p_pred)
        v_gt, v_pred = torch.Tensor([.5]), torch.Tensor([-.2])
        l = nn.loss(states, (p_pred, v_pred), (p_gt, v_gt)).detach().numpy().reshape(-1)[0]

        mse_loss = .49
        ce_loss = .9514
        self.assertAlmostEqual(mse_loss + ce_loss, l, places=4)


    def test_combined_loss(self):
        gi = TwoPlayerGuessIt()
        nn = NeuralNetwork(gi, DumbNet)
        init = gi.get_initial_state()
        template = np.zeros_like(gi.get_available_actions(init))
        template[0, 1] = 1
        s = gi.take_action(init, template)
        states = np.array([init, s])

        p_gt_top = np.ones(4) * .1
        p_gt_top[-1] = .7
        p_gt_bot = np.ones(3) * .2
        p_gt_bot[-1] = .6
        p_gt = np.array([p_gt_top, p_gt_bot], dtype=np.object)
        p_pred = np.ones((2,2,2), dtype=np.float32)
        p_pred[:,1,1] = 2
        p_pred = torch.from_numpy(p_pred)
        v_gt, v_pred = torch.Tensor([1., .5]), torch.Tensor([-.35, -.2])
        l = nn.loss(states, (p_pred, v_pred), (p_gt, v_gt)).detach().numpy().reshape(-1)[0]

        self.assertAlmostEqual((.49 + .9514) + (1.8225 + 1.0437), l, places=4)
        


if __name__ == '__main__':
    unittest.main()