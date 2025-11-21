import numpy as np

# -------------------------
# TSP with Hopfield Network
# -------------------------

N = 10  # number of cities

# Random symmetric distance matrix using random 2D coordinates
rng = np.random.default_rng(0)
coords = rng.random((N, 2))  # random 2D coordinates
dist = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i != j:
            dist[i, j] = np.linalg.norm(coords[i] - coords[j])
        else:
            dist[i, j] = 0.0

# Number of neurons = N cities * N positions
num_neurons = N * N

def idx(city, pos):
    """Map (city, position) -> 1D neuron index."""
    return city * N + pos

# -------------------------
# Build weight matrix W and bias I
# -------------------------
A = 3.0  # one city per position constraint
B = 3.0  # one position per city constraint
C = 1.0  # distance term
D = 3.0  # bias term to push neurons ON

W = np.zeros((num_neurons, num_neurons))
I = np.zeros(num_neurons)

for X in range(N):
    for i in range(N):
        p = idx(X, i)

        # Bias term (same for all neurons): encourages activity
        I[p] = D

        for Y in range(N):
            for j in range(N):
                q = idx(Y, j)

                # same city, different position -> penalize
                if X == Y and i != j:
                    W[p, q] += -A

                # same position, different city -> penalize
                if i == j and X != Y:
                    W[p, q] += -B

                # distance term: neighbors in tour (positions i+1 and i-1)
                if (j == (i + 1) % N) or (j == (i - 1) % N):
                    W[p, q] += -C * dist[X, Y]

# No self-connections
np.fill_diagonal(W, 0.0)

print("Number of neurons:", num_neurons)
print("Number of distinct weights (symmetric, no self):", num_neurons*(num_neurons-1)//2)

# -------------------------
# Hopfield dynamics (binary neurons {0,1})
# -------------------------

def run_hopfield(W, I, steps=500):
    num_neurons = W.shape[0]
    # Random initial state
    state = (rng.random(num_neurons) > 0.5).astype(int)

    for step in range(steps):
        prev = state.copy()
        # Asynchronous update in random order
        for i in rng.permutation(num_neurons):
            h = np.dot(W[i], state) + I[i]
            state[i] = 1 if h > 0 else 0

        if np.array_equal(state, prev):
            # reached a stable state
            break

    return state

state = run_hopfield(W, I, steps=500)

# -------------------------
# Decode state into tour
# -------------------------
grid = state.reshape(N, N)  # grid[city, position]

print("Final neuron grid (city x position):")
print(grid)

# For each position, pick the city with maximum activity
tour = []
for pos in range(N):
    col = grid[:, pos]
    if col.max() == 0:
        tour.append(None)
    else:
        tour.append(int(np.argmax(col)))

print("Decoded tour (city index at each position):", tour)
print("Unique cities used:", set(tour))

# Compute tour length if valid
if None not in tour and len(set(tour)) == N:
    length = 0.0
    for k in range(N):
        c1 = tour[k]
        c2 = tour[(k + 1) % N]
        length += dist[c1, c2]
    print("Valid tour found. Length =", length)
else:
    print("Tour is not a valid permutation of cities.")
