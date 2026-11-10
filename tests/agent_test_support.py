"""Load actual agent method bodies without importing simulator/model backends.

Tests provide doubles for perception, mapping and planning. The agent's control
flow itself is compiled unchanged from PSG_Nav.py, including the entire act().
"""
import ast
import copy
from pathlib import Path

import numpy as np
import torch

from utils.log_utils import log_event, vprint


def load_agent_methods(*names, path=None):
    path = Path(path) if path else Path(__file__).resolve().parents[1] / 'PSG_Nav.py'
    tree = ast.parse(path.read_text(), filename=str(path))
    agent = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == 'PSG_Nav_Agent')
    methods = [node for node in agent.body
               if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {method.name for method in methods} == set(names)
    agent.body = methods
    module = ast.Module(body=[agent], type_ignores=[])
    namespace = dict(np=np, torch=torch, copy=copy, log_event=log_event, vprint=vprint)
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace['PSG_Nav_Agent']
