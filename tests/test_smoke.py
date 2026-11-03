"""Run `pesto -j` on 2PTC, whose author chains E, I are label chains A, B, and check the tracks."""

import os
import sys
import json
import shutil
import subprocess

import gemmi
import pytest


fixture = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures', '2ptc.cif')
interface_names = ['protein', 'nucleic', 'ion', 'ligand', 'lipid']


@pytest.fixture(scope='module')
def out_dir(tmp_path_factory):
    # the console script of the environment the tests run in, not any pesto on the PATH
    pesto = shutil.which('pesto', path=os.path.dirname(sys.executable))
    assert pesto is not None, "pesto is not installed in this environment"

    out_dir = tmp_path_factory.mktemp('pesto')
    subprocess.run([pesto, '-j', '-d', 'cpu', '-o', str(out_dir), fixture], check=True)
    return out_dir


def author_residues():
    # (chain, number, insertion code) of every residue of the first model, water left out
    st = gemmi.read_structure(fixture)
    return {(chain.name, residue.seqid.num, residue.seqid.icode.strip())
            for chain in st[0] for residue in chain if not residue.is_water()}


def test_fixture_chains_differ():
    # the test is only meaningful if author and label chains differ
    st = gemmi.read_structure(fixture)
    assert {chain.name for chain in st[0]} == {'E', 'I'}
    assert {residue.subchain for residue in st[0]['E']}.isdisjoint({'E', 'I'})


@pytest.mark.parametrize('name', interface_names)
def test_track(out_dir, name):
    with open(os.path.join(out_dir, f"2ptc_pesto_{name}.json"), encoding='utf-8') as fs:
        track = json.load(fs)

    # required fields and the scale of a probability
    assert track['schema_version'] == 1
    assert track['id'] == f"pesto-i-v4-1-{name}"
    assert track['label']
    assert track['index'] == "auth"
    assert track['scale']['kind'] == "sequential"
    assert track['scale']['domain'] == [0.0, 1.0]
    assert track['palette'] == {'name': "greens"}
    assert track['provenance']['tool'] == "pesto"
    assert track['provenance']['tool_version'] not in ("", "unknown")
    assert track['provenance']['source_structure'] == "2ptc.cif"
    assert 'structure_ref' not in track

    # one entry per residue, a probability or null
    keys = []
    for entry in track['values']:
        assert set(entry) <= {'chain', 'seq', 'ins_code', 'value'}
        assert isinstance(entry['seq'], int)
        assert entry.get('ins_code', "x") != ""
        assert (entry['value'] is None) or (0.0 <= entry['value'] <= 1.0)
        keys.append((entry['chain'], entry['seq'], entry.get('ins_code', "")))
    assert len(keys) == len(set(keys))

    # on the author chains and numbers, insertion codes included, covering the structure
    assert {key[0] for key in keys} == {'E', 'I'}
    assert set(keys) == author_residues()
    assert any(key[2] for key in keys)
    assert sum(entry['value'] is not None for entry in track['values']) > 0.9*len(keys)
