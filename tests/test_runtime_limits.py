from types import SimpleNamespace
import numpy as np


def test_agent_does_not_stop_at_old_500_step_limit():
    from PSG_Nav import PSG_Nav_Agent
    agent=PSG_Nav_Agent.__new__(PSG_Nav_Agent)
    agent.max_episode_steps=1000
    agent.total_steps=500
    class ReachedPerception(Exception): pass
    def marker(*args): raise ReachedPerception()
    agent.scenegraph=SimpleNamespace(set_agent=marker)
    import pytest
    with pytest.raises(ReachedPerception):
        agent.act({'depth':np.ones((2,2,1)), 'rgb':np.zeros((2,2,3),dtype=np.uint8)})
    assert agent.total_steps==501
    agent.total_steps=1000
    assert agent.act({})=={'action':0}


def test_extra_turn_action_registered_without_replacing_global_habitat_sim():
    import habitat
    from habitat.sims.habitat_simulator.actions import HabitatSimV1ActionSpaceConfiguration
    config=habitat.get_config('configs/instancenav_val20_qwen.yaml')
    actions=HabitatSimV1ActionSpaceConfiguration(config.SIMULATOR).get()
    assert actions[6].name=='turn_right'
    assert actions[6].actuation.amount==60.0
