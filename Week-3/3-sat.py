import random

#3-SAT
def generate_3sat(m, n):
    """Generate random 3-SAT instance with m clauses, n variables"""
    clauses = []
    for _ in range(m):
        vars3 = random.sample(range(1, n+1), 3)  
        clause = []
        for v in vars3:
            if random.choice([True, False]):
                clause.append(v)   
            else:
                clause.append(-v)  
        clauses.append(clause)
    return clauses

def evaluate(clauses, assignment):
    """Number of satisfied clauses"""
    count = 0
    for c in clauses:
        ok = False
        for lit in c:
            v = abs(lit)
            val = assignment[v]
            if (lit > 0 and val) or (lit < 0 and not val):
                ok = True
        if ok: count += 1
    return count

# two heuristics
def h1(clauses, A): return evaluate(clauses, A)
def h2(clauses, A): return 2*evaluate(clauses, A) - len(clauses)

# hill climbing
def hill_climbing(clauses, n, heuristic, steps=1000):
    A = [None] + [random.choice([False, True]) for _ in range(n)]
    for _ in range(steps):
        if evaluate(clauses, A) == len(clauses):
            return True
        
        best_score, best_var = heuristic(clauses, A), None
        for v in range(1, n+1):
            A[v] = not A[v]
            score = heuristic(clauses, A)
            A[v] = not A[v]
            if score > best_score:
                best_score, best_var = score, v
        if best_var is None:  
            v = random.randint(1, n)
            A[v] = not A[v]
        else:
            A[best_var] = not A[best_var]
    return False

def beam_search(clauses, n, heuristic, beam_width=3, steps=200):
    beam = []
    for _ in range(beam_width):
        A = [None] + [random.choice([False, True]) for _ in range(n)]
        beam.append(A)
    for _ in range(steps):
        new_beam = []
        for A in beam:
            if evaluate(clauses, A) == len(clauses): return True
            for v in range(1, n+1):
                B = A.copy()
                B[v] = not B[v]
                new_beam.append(B)
        new_beam.sort(key=lambda x: heuristic(clauses, x), reverse=True)
        beam = new_beam[:beam_width]
    return False

def vnd(clauses, n, heuristic, steps=1000):
    A = [None] + [random.choice([False, True]) for _ in range(n)]
    for _ in range(steps):
        if evaluate(clauses, A) == len(clauses): return True
        improved = False
        for v in range(1, n+1):
            A[v] = not A[v]
            if heuristic(clauses, A) > heuristic(clauses, A):
                improved = True
            A[v] = not A[v]
        if not improved:   
            v = random.randint(1, n)
            A[v] = not A[v]
    return False


n = 6   # variables
m = 15  # clauses
clauses = generate_3sat(m, n)
print("Generated 3-SAT clauses:", clauses)

for hname, hfun in [("h1", h1), ("h2", h2)]:
    print(f"\nUsing heuristic {hname}:")
    print(" Hill-Climbing:", hill_climbing(clauses, n, hfun))
    print(" Beam Search (width=3):", beam_search(clauses, n, hfun, beam_width=3))
    print(" Beam Search (width=4):", beam_search(clauses, n, hfun, beam_width=4))
    print(" VND:", vnd(clauses, n, hfun))
