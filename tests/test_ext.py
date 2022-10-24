import numpy as np
from gwarell import Sim


def test_sim():
    s = Sim("simi")
    s.run(3)
    print(s.m)
