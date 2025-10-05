import random

def generate_kSAT(k, m, n):
    clauses = []
    for _ in range(m):
        # pick k distinct variables
        vars_chosen = random.sample(range(1, n+1), k)
        clause = []
        for v in vars_chosen:
            if random.choice([True, False]):
                clause.append(f"x{v}")
            else:
                clause.append(f"¬x{v}")
        clauses.append(clause)
    return clauses

# Example usage
k = 3   # length of each clause
m = 5   # number of clauses
n = 4   # number of variables

formula = generate_kSAT(k, m, n)

print("Random k-SAT formula:")
for i, clause in enumerate(formula, 1):
    print(f"C{i}: ({' ∨ '.join(clause)})")
