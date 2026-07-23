import os
import sys
from pathlib import Path

import torch

# Ensure phi_agents is in path
sys.path.append(str(Path().resolve()))

from phi_agents.appworld.interface import AppWorldInterface
from phi_agents.rl.appworld_scenario_runner import compute_turn_token_spans
from phi_agents.rl.type_defs import PolicyMessage
from phi_agents.rl.vcc.credit_weights import compute_credit_weights


def test_credit_weights():
    print("Running test_credit_weights...")
    turn_bookmarks = [False, True, False]
    turn_token_spans = [(0, 10), (10, 30), (30, 40)]

    weights = compute_credit_weights(turn_bookmarks, turn_token_spans, alpha=0.1)

    assert weights.shape[0] == 40
    assert torch.allclose(weights[10:30], torch.tensor(1.0 * 40 / 22))
    assert torch.allclose(weights[0:10], torch.tensor(0.1 * 40 / 22))
    assert torch.allclose(weights[30:40], torch.tensor(0.1 * 40 / 22))
    print("✓ compute_credit_weights test passed!")


def test_turn_token_spans():
    print("Running test_turn_token_spans...")
    messages = [
        PolicyMessage(
            "hello",
            prompt_tokens=[0],
            stopped_by_max_tokens_limit=False,
            generated_tokens=[1, 2, 3],
            generated_token_logprobs=[-0.1, -0.2, -0.3],
        ),
        PolicyMessage(
            "world",
            prompt_tokens=[0],
            stopped_by_max_tokens_limit=False,
            generated_tokens=[4, 5],
            generated_token_logprobs=[-0.4, -0.5],
        ),
    ]
    spans = compute_turn_token_spans(messages)
    assert spans == [(0, 3), (3, 5)]
    print("✓ compute_turn_token_spans test passed!")


def test_execute_with_bookmark():
    print("Running test_execute_with_bookmark...")
    os.environ["APPWORLD_ROOT"] = "/home/prathapsk/Postgraduate-Dissertation/Code/ml-loop/data/appworld_root"
    print("Initializing AppWorldInterface...")
    world = AppWorldInterface(stdout_to_devnull=True)
    try:
        task_id = "82e2fac_1"
        world.initialize(task_id, "test_vcc", raise_on_unsafe_syntax=False)

        print("Executing read API/comment...")
        code = "print('Hello world')"
        output, is_bookmark = world.execute_with_bookmark(code)
        print(f"Read API code output: {output}, is_bookmark={is_bookmark}")
        assert is_bookmark is False, f"Expected comment to not be a bookmark but got {is_bookmark}"

        print("Executing state-changing API...")
        # complete_task is a standard write API call in AppWorld supervisor app
        code = "apis.supervisor.complete_task()"
        output, is_bookmark = world.execute_with_bookmark(code)
        print(f"Write API code output: {output}, is_bookmark={is_bookmark}")
        assert is_bookmark is True, f"Expected complete_task to be a bookmark but got {is_bookmark}"

        print("✓ execute_with_bookmark test passed!")

    finally:
        world.close_server()


if __name__ == "__main__":
    test_credit_weights()
    test_turn_token_spans()
    test_execute_with_bookmark()
