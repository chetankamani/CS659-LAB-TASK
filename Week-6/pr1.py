import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        """
        size: number of neurons (for 10x10 image -> size = 100)
        """
        self.size = size
        self.W = np.zeros((size, size))

    @staticmethod
    def bin_to_bipolar(pattern):
        """
        Convert binary {0,1} pattern to bipolar {-1,+1}
        """
        return np.where(pattern == 0, -1, 1)

    @staticmethod
    def bipolar_to_bin(pattern):
        """
        Convert bipolar {-1,+1} pattern to binary {0,1}
        """
        return np.where(pattern <= 0, 0, 1)

    def train(self, patterns):
        """
        Train the Hopfield network on a list/array of patterns.
        patterns: list/array of vectors of shape (size,) with values in {0,1} or {-1, +1}
        """
        self.W = np.zeros((self.size, self.size))

        for p in patterns:
            p = np.array(p)
            # Ensure bipolar
            if set(np.unique(p)) <= {0, 1}:
                p = self.bin_to_bipolar(p)
            p = p.reshape(-1, 1)
            self.W += p @ p.T

        # Zero out diagonal
        np.fill_diagonal(self.W, 0)

        # Optionally normalize by number of patterns
        self.W /= len(patterns)

    def recall(self, pattern, steps=10, synchronous=True):
        """
        Recall a pattern from the network.
        pattern: vector (size,) with values in {0,1} or {-1,+1}
        steps: number of update iterations
        synchronous: if True, update all neurons at once; else asynchronous
        """
        state = np.array(pattern, dtype=float)

        # Ensure bipolar
        if set(np.unique(state)) <= {0, 1}:
            state = self.bin_to_bipolar(state)

        for _ in range(steps):
            if synchronous:
                state = np.sign(self.W @ state)
                # handle zeros from sign (keep previous value)
                zero_idx = (state == 0)
                state[zero_idx] = pattern[zero_idx]
            else:
                # asynchronous update: random order
                for i in np.random.permutation(self.size):
                    s = np.dot(self.W[i, :], state)
                    if s > 0:
                        state[i] = 1
                    elif s < 0:
                        state[i] = -1
                    # if s == 0, keep previous state[i]

        return state

    def energy(self, state):
        """
        Compute Hopfield energy of current state.
        """
        state = np.array(state, dtype=float)
        if set(np.unique(state)) <= {0, 1}:
            state = self.bin_to_bipolar(state)
        return -0.5 * state.T @ self.W @ state


# =========================
# Example usage: 10x10 images
# =========================
def pattern_to_vector(pattern_2d):
    """Flatten 10x10 binary pattern to length-100 vector"""
    return np.array(pattern_2d).reshape(-1)

def vector_to_pattern(vec, shape=(10, 10)):
    """Reshape vector back to 10x10 pattern"""
    return np.array(vec).reshape(shape)

if __name__ == "__main__":
    # Define two simple 10x10 binary patterns (e.g., X and +)
    P1 = np.zeros((10, 10), dtype=int)
    for i in range(10):
        P1[i, i] = 1
        P1[i, 9 - i] = 1

    P2 = np.zeros((10, 10), dtype=int)
    P2[4:6, :] = 1
    P2[:, 4:6] = 1

    p1_vec = pattern_to_vector(P1)
    p2_vec = pattern_to_vector(P2)

    # Create Hopfield network for 10x10 = 100 neurons
    hop = HopfieldNetwork(size=100)

    # Train on both patterns
    hop.train([p1_vec, p2_vec])

    # Create a noisy version of P1
    noisy_p1 = p1_vec.copy()
    noise_indices = np.random.choice(100, size=15, replace=False)
    noisy_p1[noise_indices] = 1 - noisy_p1[noise_indices]  # flip bits

    # Recall from noisy pattern
    recalled_vec = hop.recall(noisy_p1, steps=10, synchronous=True)
    recalled_bin = hop.bipolar_to_bin(recalled_vec)
    recalled_pattern = vector_to_pattern(recalled_bin)

    # Print results
    print("Original P1:")
    print(P1)
    print("Noisy P1:")
    print(vector_to_pattern(noisy_p1))
    print("Recalled P1:")
    print(recalled_pattern)
