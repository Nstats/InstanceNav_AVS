from types import SimpleNamespace
import numpy as np
import torch
from utils.map_manager import MapManager


def test_unknown_room_label_is_ignored_without_crashing_or_assigning_wrong_room():
    calls=[]
    agent=SimpleNamespace(
        detection_manager=SimpleNamespace(_get_glip_real_label=lambda _:['room','living room']),
        config=SimpleNamespace(SIMULATOR=SimpleNamespace(DEPTH_SENSOR=SimpleNamespace(HEIGHT=8,WIDTH=8))),
        full_pose=None,room_map=None,
        room_map_module=lambda *args:calls.append(args) or 'updated',
    )
    manager=MapManager.__new__(MapManager);manager.agent=agent;manager.device='cpu'
    prediction=SimpleNamespace(bbox=torch.tensor([[0,0,2,2],[3,3,7,7]]),
                               get_field=lambda _:torch.tensor([.9,.8]))
    manager.update_room_map({'depth':np.ones((8,8,1),dtype=np.float32)},prediction)
    assert agent.room_map=='updated'
    mask=calls[0][3]
    assert mask[:,0:2,0:2].sum()==0
    assert mask[1,3:7,3:7].sum()==16
