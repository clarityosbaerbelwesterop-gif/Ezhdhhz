"""Process verifier for the addition demo: checks every step of a solution against the prompt,
without knowing the correct answer. Score 1 = all steps consistent, else 0 (+ partial credit)."""
import re

STEP = re.compile(r"^(\d)\+(\d)=(\d+),(\d)\+(\d)\+(\d)=(\d+):(\d+)$")


def check(prompt: str, completion: str) -> float:
    m = re.match(r"^(\d\d)\+(\d\d)=$", prompt)
    s = STEP.match(completion.strip())
    if not m or not s:
        return 0.0
    a, b = int(m.group(1)), int(m.group(2))
    u1, u2, su, t1, t2, c, st, ans = map(int, s.groups())
    ok = [u1 == a % 10, u2 == b % 10, su == u1 + u2, t1 == a // 10, t2 == b // 10, c == su // 10,
          st == t1 + t2 + c, ans == st * 10 + su % 10]
    return 1.0 if all(ok) else sum(ok) / len(ok) / 2
