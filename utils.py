# utils.py -- small helper functions
import math

def euclid(a: tuple, b: tuple) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

def dbm_to_mw(dbm: float) -> float:
    return 10 ** (dbm / 10.0)

def mw_to_dbm(mw: float) -> float:
    return 10.0 * math.log10(mw) if mw > 0 else float("-inf")
