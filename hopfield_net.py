"""
Simple Discrete Hopfield Net

The Hopfield network is designed to store a number of patterns so that they can
be retrieved from noisy or partial cues. They can also be used to determine
whether an input vector is a "known" or an "unknown" vector
(Content-Addressable Memory). This patterns are stored in a weight matrix.

The net has symmetric weights with no self-connections i.e w_ij = w_ji and
w_ii=0. Only one unit updates its activation at a time.

The asynchronous updating of the units allows a function, known as the energy
or Lyapunov function, to be found for the net that proves the net will
converge to a stable set of activations, rather than oscillating.

References:
https://en.wikipedia.org/wiki/Hopfield_network
http://staff.itee.uq.edu.au/janetw/cmc/chapters/Hopfield/
http://web.cs.ucla.edu/~rosen/161/notes/hopfield.html
"""
import numpy as np
from random import shuffle


class Hopfield:
    def __init__(self, neuron_num=4, threshold=0):
        self.neuron_num = neuron_num
        self.threshold = threshold

        # Weight Matric used to store patterns
        self.weight_matrix = np.zeros((self.neuron_num, self.neuron_num))

    def update_net(self, input_vector):
        """
        Update each node randomly until convergence
        v_in = sum(w_ij * v_j)
        if v_in >= threshold, v_in=1
        else v_in=1
        """
        convergence = False
        intermediate_out = input_vector.copy().reshape(-1)
        node_index = list(range(self.neuron_num))

        while not convergence:
            shuffle(node_index)
            changed = False
            for node in node_index:
                val = input_vector.reshape(-1)[node] + np.sum(
                    self.weight_matrix[node].reshape(1, -1) * intermediate_out)
                if val >= self.threshold and intermediate_out[node] != 1:
                    intermediate_out[node] = 1
                    changed = True
                elif val < self.threshold and intermediate_out[node] != 0:
                    intermediate_out[node] = 0
                    changed = True

            if not changed:
                convergence = True

        return intermediate_out

    def train_net(self, input_vector):
        """
        Use formula to update weight matrix for node:
        w_ij = sum( (2*V_i - 1)(2*V_j - 1) )
        """
        new_weight_matrix = (2 * input_vector - 1) * (2 * input_vector.T - 1)
        new_weight_matrix = np.sum(new_weight_matrix, axis=1)
        new_weight_matrix = (1 - np.eye(self.neuron_num)) * new_weight_matrix
        self.weight_matrix = new_weight_matrix


def main():
    input_vector = np.array([[[1, 1, 1, 0]]])
    noisy_input = np.array([[0, 0, 1, 0]])

    hopfield_net = Hopfield(neuron_num=4, threshold=0)
    hopfield_net.train_net(input_vector)
    reconstructed_input = hopfield_net.update_net(noisy_input)
    print("Noisy Input: ", noisy_input, " Reconstructed Input: ", reconstructed_input)


if __name__ == '__main__':
    main()