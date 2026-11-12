"""RL reward for one rollout: did the job verifiably succeed, and without wasted steps.

Binary on purpose: the spec check (teacher.verify) is all-or-nothing, so partial credit can't be farmed by producing
some of the outputs or a file that only looks right. Passing jobs lose a little for steps beyond the reference
solution plus slack, so the model learns to stop fumbling, never enough to make a pass worth less than half.
"""
SLACK = 2        # free steps beyond the reference (a probe or a retry is fine)
STEP_COST = 0.05
FLOOR = 0.5


def ref_steps(row):
    """Commands in the reference solution, not counting a deliberate wrong first attempt (weight 0)."""
    return sum(1 for m in row["messages"] if m["role"] == "assistant" and m.get("tool_calls") and m.get("weight", 1))


def reward(result, row):
    """result: eval_agent.run_job output ({"pass", "steps", ...}); row: the task row it ran."""
    if not result["pass"]:
        return 0.0
    extra = max(0, result["steps"] - ref_steps(row) - SLACK)
    return max(FLOOR, 1.0 - STEP_COST * extra)


if __name__ == "__main__":
    row = {"messages": [{"role": "system"}, {"role": "user"},
                        {"role": "assistant", "tool_calls": [{}], "weight": 0}, {"role": "tool"},  # wrong attempt
                        {"role": "assistant", "tool_calls": [{}]}, {"role": "tool"},
                        {"role": "assistant", "tool_calls": [{}]}, {"role": "tool"},
                        {"role": "assistant", "content": "Done."}]}
    assert ref_steps(row) == 2
    assert reward({"pass": False, "steps": 2}, row) == 0.0
    assert reward({"pass": True, "steps": 2}, row) == 1.0
    assert reward({"pass": True, "steps": 4}, row) == 1.0           # within slack
    assert abs(reward({"pass": True, "steps": 6}, row) - 0.9) < 1e-9
    assert reward({"pass": True, "steps": 40}, row) == FLOOR
    assert reward({"pass": True, "steps": 0}, {"messages": [{"role": "assistant", "content": "Which file?"}]}) == 1.0
    print("reward: all checks passed")
