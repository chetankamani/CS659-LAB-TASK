import numpy as np

# Board size
N = 8  # 8x8 board -> 64 neurons

def idx(i, j, N=8):
    """Map 2D (row, col) to 1D neuron index."""
    return i * N + j

def build_eight_rooks_hopfield(N=8, A=1.0, B=1.0):
    """
    Build weight matrix W and thresholds theta for the Eight-rook problem.
    
    Energy:
        E = A * sum_i (sum_j x_ij - 1)^2 + B * sum_j (sum_i x_ij - 1)^2
    """
    num_neurons = N * N
    W = np.zeros((num_neurons, num_neurons))
    theta = np.zeros(num_neurons)

    # Row constraints: penalize more than one rook in a row
    # Add pairwise terms: for each row i, for each pair (j1, j2), j1 != j2
    for i in range(N):
        for j1 in range(N):
            for j2 in range(N):
                if j1 == j2:
                    continue
                p = idx(i, j1, N)
                q = idx(i, j2, N)
                # Coefficient in energy is +2A * x_p x_q -> w_pq = -2A
                W[p, q] += -2 * A

    # Column constraints: penalize more than one rook in a column
    for j in range(N):
        for i1 in range(N):
            for i2 in range(N):
                if i1 == i2:
                    continue
                p = idx(i1, j, N)
                q = idx(i2, j, N)
                # Coefficient in energy is +2B * x_p x_q -> w_pq = -2B
                W[p, q] += -2 * B

    # Thresholds from linear terms: coefficient is -(A+B) * x_ij
    # In Hopfield energy: E = ... + sum theta_p * x_p
    # so theta_p = -(A+B)
    theta[:] = -(A + B)

    # No self-connections
    np.fill_diagonal(W, 0.0)

    return W, theta


def hopfield_run(W, theta, steps=100, initial_state=None):
    """
    Asynchronous Hopfield update for binary neurons {0,1}.
    s_i(t+1) = 1 if sum_j W_ij s_j > theta_i else 0
    """
    num_neurons = W.shape[0]
    if initial_state is None:
        state = (np.random.rand(num_neurons) > 0.5).astype(int)
    else:
        state = np.array(initial_state, dtype=int)

    for _ in range(steps):
        prev = state.copy()
        # Asynchronous update in random order
        for i in np.random.permutation(num_neurons):
            h = np.dot(W[i], state)
            state[i] = 1 if h > theta[i] else 0

        if np.array_equal(state, prev):
            break  # reached a stable state

    return state


def energy(W, theta, state):
    """Compute Hopfield energy for binary state {0,1}."""
    s = state.astype(float)
    return -0.5 * s @ W @ s + np.dot(theta, s)


def print_board(state, N=8):
    """Print the board: R for rook (1), . for empty (0)."""
    board = state.reshape(N, N)
    for i in range(N):
        print(" ".join('R' if board[i, j] == 1 else '.' for j in range(N)))
    print()


def is_valid_eight_rooks(state, N=8):
    """Check if state is a valid eight-rook solution."""
    board = state.reshape(N, N)
    row_sums = board.sum(axis=1)
    col_sums = board.sum(axis=0)
    return np.all(row_sums == 1) and np.all(col_sums == 1)


if __name__ == "__main__":
    N = 8
    A = 1.0
    B = 1.0

    # Build Hopfield network for eight-rook problem
    W, theta = build_eight_rooks_hopfield(N=N, A=A, B=B)

    # Try several random initial states until we find a valid solution
    num_tries = 20
    solution = None

    for attempt in range(num_tries):
        init_state = (np.random.rand(N * N) > 0.5).astype(int)
        final_state = hopfield_run(W, theta, steps=200, initial_state=init_state)

        if is_valid_eight_rooks(final_state, N=N):
            solution = final_state
            print(f"Found valid solution on attempt {attempt + 1}")
            break

    if solution is not None:
        print("Final board configuration:")
        print_board(solution, N=N)
        print("Energy:", energy(W, theta, solution))
    else:
        print("No valid solution found in the given number of attempts.")
