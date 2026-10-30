from pathlib import Path
import numpy as np
from tools.progress_snapshots import inspect,outcar_frames,capture

def structure(distance):
 return {'atoms':['Cu','Cu','C','C','O','O','O','H','H'],'positions':np.array([[0,0,0],[distance,0,0],[0,0,2],[distance,0,2],[0,0,3.16],[distance,0,3.16],[7,7,3],[7.9,7,3],[6.7,7.8,3.]]),'cell':np.diag([20.,20.,20.])}

def test_water_is_not_protonation_and_bond_is_not_inferred_from_name():
 d=inspect(structure(3.3),[2,3],[4,5]);assert d['verdict']=='dissociated 2CO';assert not d['protonated_dimer_candidates']
 d=inspect(structure(1.5),[2,3],[4,5]);assert d['verdict']=='retained two-C-bound OCCO'

def test_partial_force_block_is_not_a_complete_ionic_step(tmp_path):
 s={'atoms':['C','C'],'positions':np.array([[0,0,0],[1.5,0,0]]),'cell':np.eye(3)*10};p=tmp_path/'OUTCAR';p.write_text(' POSITION TOTAL-FORCE\n ----\n 0 0 0 0 0 0\n 1.5 0 0 0 0 0\n POSITION TOTAL-FORCE\n ----\n 0 0 0 0 0 0\n')
 frames,forces,steps=outcar_frames(p,s);assert len(frames)==1 and steps==[1]

def test_stable_snapshot_preserves_source_bytes(tmp_path):
 p=tmp_path/'live';p.write_bytes(b'checkpoint\n');r=capture(p,tmp_path/'snapshot');assert r['stable_read'] and p.read_bytes()==(tmp_path/'snapshot').read_bytes()
