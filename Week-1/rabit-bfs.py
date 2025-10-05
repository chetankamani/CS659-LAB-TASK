from collections import deque

def next_states(state):
    s = list(state)
    i = s.index('_')
    if i - 1 >= 0 and s[i - 1] == '>':
        t = s.copy(); t[i], t[i - 1] = t[i - 1], t[i]
        yield ('> moves 1 right', ''.join(t))
    if i + 1 < len(s) and s[i + 1] == '<':
        t = s.copy(); t[i], t[i + 1] = t[i + 1], t[i]
        yield ('< moves 1 left', ''.join(t))
    if i - 2 >= 0 and s[i - 2] == '>' and s[i - 1] == '<':
        t = s.copy(); t[i], t[i - 2] = t[i - 2], t[i]
        yield ('> jumps over <', ''.join(t))
    if i + 2 < len(s) and s[i + 2] == '<' and s[i + 1] == '>':
        t = s.copy(); t[i], t[i + 2] = t[i + 2], t[i]
        yield ('< jumps over >', ''.join(t))

def bfs(start=">>>_<<<", goal="<<<_>>>"):
    q = deque([start])
    parent = {start: None}   
    action = {}              
    seen = {start}
    while q:
        u = q.popleft()
        if u == goal:
            break
        for a, v in next_states(u):
            if v not in seen:
                seen.add(v)
                parent[v] = u
                action[v] = a
                q.append(v)
    if goal not in parent:
        return None
    path = []
    s = goal
    while s is not None:
        path.append(s)
        s = parent[s]
    path.reverse()
    steps = []
    for i in range(1, len(path)):
        steps.append((action[path[i]], path[i]))
    return steps


steps = bfs()
print("BFS found steps:", len(steps))
for move, state in steps:
    print(f"{move:>18} -> {state}")
