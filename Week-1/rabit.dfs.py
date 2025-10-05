# Depth First Search for Rabbit Leap problem

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

def dfs(start=">>>_<<<", goal="<<<_>>>"):
    stack = [(start, [])]
    visited = {start}
    while stack:
        state, path = stack.pop()
        if state == goal:
            return path
        for move, nxt in reversed(list(next_states(state))): 
            if nxt not in visited:
                visited.add(nxt)
                stack.append((nxt, path + [(move, nxt)]))
    return None


steps = dfs()
print("DFS found steps:", len(steps))
for move, state in steps:
    print(f"{move:>18} -> {state}")
