#!/usr/bin/env python3
"""Apply a trained PeSTo model to PDB/mmCIF files.

Writes the predicted interface probabilities into the b-factor field of a copy
of the input structure (PDB or mmCIF, chain names, residue numbering and
insertion codes preserved), into a per-residue table (TSV/CSV) and, with --json,
into one residue score track per interface type.

All chains found in one input file are predicted together as a single unit,
like in apply_model.ipynb. Supply isolated subunits if that is what you want
predictions for.
"""

import os
import re
import sys
import gzip
import json
import hashlib
import argparse
import importlib.util
import numpy as np
import torch as pt
from glob import glob
from tqdm import tqdm
from importlib import metadata
from datetime import datetime, timezone

import gemmi

from .dataset import collate_batch_features
from .data_encoding import encode_structure, encode_features
from .structure import clean_structure, tag_hetatm_chains, split_by_chain
from .structure import filter_non_atomic_subunits, remove_duplicate_tagged_subunits, concatenate_chains


# interface types predicted by the model, in the order of the output channels
# (see 'r_types' in the model config)
interface_names = ['protein', 'nucleic', 'ion', 'ligand', 'lipid']
interface_labels = {
    'protein': "protein interface",
    'nucleic': "nucleic acid interface",
    'ion': "ion binding site",
    'ligand': "ligand binding site",
    'lipid': "lipid binding site",
}

# the trained models, each with the configuration and architecture it was trained with
weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')
model_dirs = {
    'i_v3_0': "i_v3_0_2021-05-27_14-27",
    'i_v3_1': "i_v3_1_2021-05-28_12-40",
    'i_v4_0': "i_v4_0_2021-09-07_11-20",
    'i_v4_1': "i_v4_1_2021-09-07_11-21",
}

# structure formats we read and write ('.pdb1', '.pdb2', ... are bioassemblies)
re_pdb = re.compile(r"\.(pdb|ent)[0-9]*(\.gz)?$", re.IGNORECASE)
re_cif = re.compile(r"\.(cif|mmcif)(\.gz)?$", re.IGNORECASE)


def package_version():
    # the version of the installed package, derived from git when it was built
    try:
        return metadata.version('pesto')
    except metadata.PackageNotFoundError:
        return "unknown"


def load_module(filepath, name):
    # import a script by its path, under a name of its own, without touching sys.path
    spec = importlib.util.spec_from_file_location(name, filepath)
    assert (spec is not None) and (spec.loader is not None)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model(model_name, device):
    # locate model, independently of the working directory
    save_path = os.path.join(weights_dir, model_dirs[model_name])
    model_filepath = os.path.join(save_path, 'model_ckpt.pt')
    if not os.path.isfile(model_filepath):
        raise FileNotFoundError(f"no weights for model {model_name} in {save_path}")

    # load the configuration and architecture stored next to the weights
    config_model = load_module(os.path.join(save_path, 'config.py'), f"pesto_{model_name}_config").config_model
    Model = load_module(os.path.join(save_path, 'model.py'), f"pesto_{model_name}_model").Model

    # reload model
    model = Model(config_model)
    model.load_state_dict(pt.load(model_filepath, map_location=pt.device("cpu")))
    model = model.eval().to(device)

    # the v4 models only use the element of each atom, the v3 models use all features
    all_features = (config_model['em']['N0'] > 30)

    # one output channel per interface type
    names = interface_names[:config_model['dm']['N2']]

    return model, names, all_features


def read_structure(filepath):
    # read pdb or mmcif
    if re_cif.search(filepath):
        doc = gemmi.read_structure(filepath, format=gemmi.CoorFormat.Mmcif)
    elif re_pdb.search(filepath):
        doc = gemmi.read_pdb(filepath, max_line_length=80)
    else:
        doc = gemmi.read_structure(filepath)

    # altloc memory
    altloc_l = []

    # data storage
    atom_element = []
    atom_name = []
    atom_xyz = []
    residue_name = []
    seq_id = []
    het_flag = []
    chain_name = []
    icodes = []

    # location of every atom in the gemmi structure, to map the predictions back
    location = []

    # parse structure
    for mid, model in enumerate(doc):
        for cid, chain in enumerate(model):
            for rid, residue in enumerate(chain):
                for atom in residue:
                    # altloc check (keep first encountered)
                    if atom.has_altloc():
                        key = f"{chain.name}_{residue.seqid.num}_{atom.name}"
                        if key in altloc_l:
                            continue
                        else:
                            altloc_l.append(key)

                    # store data
                    atom_element.append(atom.element.name)
                    atom_name.append(atom.name)
                    atom_xyz.append([atom.pos.x, atom.pos.y, atom.pos.z])
                    residue_name.append(residue.name)
                    seq_id.append(residue.seqid.num)
                    het_flag.append(residue.het_flag)
                    chain_name.append(f"{chain.name}:{mid}")
                    icodes.append(residue.seqid.icode.strip())
                    location.append([mid, cid, rid])

    # pack data
    structure = {
        'xyz': np.array(atom_xyz, dtype=np.float32),
        'name': np.array(atom_name),
        'element': np.array(atom_element),
        'resname': np.array(residue_name),
        'resid': np.array(seq_id, dtype=np.int32),
        'het_flag': np.array(het_flag),
        'chain_name': np.array(chain_name),
        'icode': np.array(icodes),
        'location': np.array(location, dtype=np.int32),
    }

    return doc, structure


def preprocess(structure):
    # process structure
    structure = clean_structure(structure)

    # update molecules chains
    structure = tag_hetatm_chains(structure)

    # split structure
    subunits = split_by_chain(structure)

    # remove non atomic structures
    subunits = filter_non_atomic_subunits(subunits)

    # remove duplicated molecules and ions
    subunits = remove_duplicate_tagged_subunits(subunits)

    # concatenate all remaining chains together
    return concatenate_chains(subunits)


def extract_topology(X, num_nn, chunk_size=1024):
    # same nearest neighbors as pesto.data_encoding.extract_topology, but without ever
    # holding the full distance matrix in memory
    knn = min(num_nn, X.shape[0])
    ids_topk = pt.zeros((X.shape[0], knn), dtype=pt.long, device=X.device)

    # no distance can be larger than the diagonal of the bounding box
    d_max = pt.norm(pt.max(X, dim=0).values - pt.min(X, dim=0).values)

    for i in range(0, X.shape[0], chunk_size):
        # compute distances of the chunk to all atoms
        D = pt.norm(X[i:i+chunk_size].unsqueeze(1) - X.unsqueeze(0), dim=2)

        # mask self and overlapping atoms
        D = D + d_max*(D < 1e-2).float()

        # find nearest neighbors
        ids_topk[i:i+chunk_size] = pt.topk(D, knn, dim=1, largest=False).indices

    return ids_topk


def predict(model, structure, device, all_features):
    # encode structure and features
    X, M = encode_structure(structure)
    if all_features:
        q = pt.cat(encode_features(structure), dim=1)
    else:
        q = encode_features(structure)[0]

    # extract topology
    ids_topk = extract_topology(X, 64)

    # pack data and setup sink (IMPORTANT)
    X, ids_topk, q, M = collate_batch_features([[X, ids_topk, q, M]])

    # run model
    with pt.no_grad():
        z = model(X.to(device), ids_topk.to(device), q.to(device), M.float().to(device))
        p = pt.sigmoid(z).cpu().numpy()

    # map each atom to its row in the predictions
    _, ids_res = np.unique(structure['resid'], return_inverse=True)

    return p, ids_res


def collect_residues(structure, p, ids_res):
    # gather the prediction of every residue of the input structure that was kept
    predictions = {}
    for (mid, cid, rid), i in zip(structure['location'], ids_res):
        key = (int(mid), int(cid), int(rid))
        if key in predictions:
            predictions[key] = np.maximum(predictions[key], p[i])
        else:
            predictions[key] = p[i]

    return predictions


def encode_bfactor(doc, predictions, i_type, default=0.0):
    # store the predictions of one interface type in the b-factor field
    for mid, model in enumerate(doc):
        for cid, chain in enumerate(model):
            for rid, residue in enumerate(chain):
                p = predictions.get((mid, cid, rid))
                bf = default if p is None else float(p[i_type])
                for atom in residue:
                    atom.b_iso = bf


def save_structure(doc, filepath):
    if re_cif.search(filepath):
        doc.setup_entities()
        doc.make_mmcif_document().write_file(filepath)
    else:
        doc.write_pdb(filepath)


def save_table(doc, predictions, names, filepath):
    # one row per residue, in the order of the input structure
    sep = ',' if filepath.endswith('.csv') else '\t'
    header = ['model', 'chain', 'resid', 'icode', 'resname'] + [f"p_{name}" for name in names]

    with open(filepath, 'w', encoding='utf-8') as fs:
        fs.write(sep.join(header)+'\n')
        for mid, model in enumerate(doc):
            for cid, chain in enumerate(model):
                for rid, residue in enumerate(chain):
                    p = predictions.get((mid, cid, rid))
                    if p is None:
                        continue

                    row = [
                        str(mid), chain.name, str(residue.seqid.num),
                        residue.seqid.icode.strip(), residue.name,
                    ] + ["{:.4f}".format(v) for v in p]
                    fs.write(sep.join(row)+'\n')


def track_residues(doc, source):
    # the residues a track covers, each with the key of its prediction. The format has no
    # model dimension, so only the first model is covered and the caller warns about the rest
    residues = []
    seen = set()
    num_duplicates = 0
    for cid, chain in enumerate(doc[0]):
        for rid, residue in enumerate(chain):
            # water is removed before the prediction and was never in scope
            if residue.is_water():
                continue

            # the chain names and residue numbers of gemmi are the author ones, its residue
            # name is auth_comp_id where an mmCIF file has one
            entry = {'chain': chain.name, 'idx': residue.seqid.num, 'res_name': residue.name}
            icode = residue.seqid.icode.strip()
            if icode:
                entry['ins_code'] = icode

            # one entry per residue
            key = (entry['chain'], entry['idx'], icode)
            if key in seen:
                num_duplicates += 1
                continue
            seen.add(key)

            residues.append((entry, (0, cid, rid)))

    if num_duplicates > 0:
        print(f"WARNING: {source}: {num_duplicates} residues share their chain, number and "
              "insertion code with an earlier one and are left out of the tracks")

    return residues


def sha1sum(filepath, chunk_size=2**20):
    # the SHA-1 of the uncompressed bytes, a gzipped file is told by its magic bytes
    with open(filepath, 'rb') as fs:
        gzipped = (fs.read(2) == b'\x1f\x8b')

    digest = hashlib.sha1()
    with (gzip.open if gzipped else open)(filepath, 'rb') as fs:
        for chunk in iter(lambda: fs.read(chunk_size), b''):
            digest.update(chunk)

    return digest.hexdigest()


def save_track(residues, predictions, names, i_type, model_name, filepath, source, source_sha1sum):
    # one residue score track, following the schema in docs/residue-score-track.md of
    # af-molstar-viewer: one scalar per residue, kept out of the structure file
    values = []
    for entry, key in residues:
        # null, never 0, for the residues dropped by the preprocessing
        p = predictions.get(key)
        values.append(dict(entry, value=None if p is None else round(float(p[i_type]), 4)))

    track = {
        'schema_version': 2,
        'id': "pesto-{}-{}".format(model_name.replace('_', '-'), names[i_type]),
        'label': "PeSTo {}".format(interface_labels[names[i_type]]),
        'index': "auth",
        'scale': {
            'kind': "sequential",
            # a probability, the range is the one of the measure and not of this structure
            'domain': [0.0, 1.0],
            'units': "probability",
        },
        # the colour the spec assigns to PeSTo, light to dark so that 0 is the light end
        'palette': {'name': "greens"},
        'provenance': {
            'tool': "pesto",
            'tool_version': package_version(),
            'created': datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            'source_structure': source,
            'source_sha1sum': source_sha1sum,
            'parameters': {
                'model': model_name,
                'weights': model_dirs[model_name],
                'interface': names[i_type],
            },
        },
        'values': [],
    }

    # keep one residue per line, indenting every entry makes the file four times longer
    entries = ",\n    ".join([json.dumps(entry, ensure_ascii=False) for entry in values])
    text = json.dumps(track, indent=2, ensure_ascii=False)
    text = text.replace('"values": []', '"values": [\n    {}\n  ]'.format(entries))

    with open(filepath, 'w', encoding='utf-8') as fs:
        fs.write(text+'\n')


def locate_structures(paths, suffix):
    # expand directories into the structure files they contain, ignoring our own results
    filepaths = []
    missing = []
    for path in paths:
        if os.path.isdir(path):
            found = [fp for fp in sorted(glob(os.path.join(path, "*")))
                     if (re_pdb.search(fp) or re_cif.search(fp)) and (suffix not in os.path.basename(fp))]
            if len(found) == 0:
                print(f"WARNING: no structure file directly in {path}, subfolders are not searched")
            filepaths.extend(found)
        elif os.path.isfile(path):
            filepaths.append(path)
        else:
            missing.append(path)

    return filepaths, missing


def output_filepath(filepath, out_dir, suffix, extension):
    # strip the (possibly double) extension of the input
    name = os.path.basename(filepath)
    m = re_pdb.search(name) or re_cif.search(name)
    name = name[:m.start()] if m is not None else os.path.splitext(name)[0]

    # keep the input next to its predictions unless asked otherwise
    directory = os.path.dirname(os.path.abspath(filepath)) if out_dir is None else out_dir

    return os.path.join(directory, name+suffix+extension)


def run(filepath, model, names, all_features, device, args):
    # read and prepare structure
    doc, structure = read_structure(filepath)
    if structure['xyz'].shape[0] == 0:
        raise ValueError("no atom found")

    structure = preprocess(structure)
    if structure['xyz'].shape[0] == 0:
        raise ValueError("nothing left to predict on")

    # the runtime scales with the square of the number of atoms
    if (args.max_atoms > 0) and (structure['xyz'].shape[0] > args.max_atoms):
        raise ValueError("{} atoms (--max-atoms is {}), skipped".format(
            structure['xyz'].shape[0], args.max_atoms))

    # run model and gather the predictions per residue of the input structure
    p, ids_res = predict(model, structure, device, all_features)
    predictions = collect_residues(structure, p, ids_res)

    # determine output format
    if args.format == 'same':
        extension = '.cif' if re_cif.search(filepath) else '.pdb'
    else:
        extension = '.'+args.format

    # store the selected interface type(s) in the b-factor field
    types = names if args.bfactor == 'all' else [args.bfactor]
    for name in types:
        suffix = args.suffix if (len(types) == 1) else f"{args.suffix}_{name}"
        encode_bfactor(doc, predictions, names.index(name))
        save_structure(doc, output_filepath(filepath, args.out_dir, suffix, extension))

    # store all interface types in a table
    if not args.no_table:
        save_table(doc, predictions, names,
                   output_filepath(filepath, args.out_dir, args.suffix, '.'+args.table_format))

    # store each interface type as a residue score track
    if args.json:
        if len(doc) > 1:
            print(f"WARNING: {filepath} has {len(doc)} models, only the first one is in the tracks")

        source = os.path.basename(filepath)
        source_sha1sum = sha1sum(filepath)
        residues = track_residues(doc, source)
        for i_type in range(len(names)):
            save_track(residues, predictions, names, i_type, args.model,
                       output_filepath(filepath, args.out_dir, f"{args.suffix}_{names[i_type]}", '.json'),
                       source, source_sha1sum)


def main():
    parser = argparse.ArgumentParser(
        prog='pesto', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('inputs', nargs='+', help="PDB/mmCIF files, or directories containing them")
    parser.add_argument('-o', '--out-dir', default=None,
                        help="where to write the results (default: next to each input file)")
    parser.add_argument('-m', '--model', default='i_v4_1', choices=sorted(model_dirs),
                        help="pre-trained model to apply (default: %(default)s)")
    parser.add_argument('-d', '--device', default='auto', help="cpu, cuda, mps, ... (default: %(default)s)")
    parser.add_argument('-f', '--format', default='same', choices=['same', 'pdb', 'cif'],
                        help="format of the output structure (default: %(default)s)")
    parser.add_argument('-b', '--bfactor', default='protein', choices=interface_names+['all'],
                        help="interface type to store in the b-factor field, 'all' writes one "
                             "structure per type (default: %(default)s)")
    parser.add_argument('-t', '--table-format', default='tsv', choices=['tsv', 'csv'],
                        help="format of the per-residue table (default: %(default)s)")
    parser.add_argument('--no-table', action='store_true', help="do not write the per-residue table")
    parser.add_argument('-j', '--json', action='store_true',
                        help="also write one residue score track per interface type, as JSON")
    parser.add_argument('--suffix', default='_pesto', help="added to the output names (default: %(default)s)")
    parser.add_argument('--max-atoms', type=int, default=30000,
                        help="skip structures with more atoms, 0 for no limit (default: %(default)s)")
    args = parser.parse_args()

    # define device
    if args.device == 'auto':
        args.device = 'cuda' if pt.cuda.is_available() else 'cpu'
    device = pt.device(args.device)

    # load model
    model, names, all_features = load_model(args.model, device)

    # check that the requested interface type is predicted by this model
    if (args.bfactor != 'all') and (args.bfactor not in names):
        parser.error("model {} only predicts {} interfaces".format(args.model, ', '.join(names)))

    # find input structures
    filepaths, missing = locate_structures(args.inputs, args.suffix)
    if len(missing) > 0:
        parser.error("no such file or directory: {}".format(', '.join(missing)))
    if len(filepaths) == 0:
        parser.error("no structure file to predict on")

    # prepare output location
    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)

    # run model on all structures, a failed one does not stop the others
    failed = []
    for filepath in tqdm(filepaths):
        try:
            run(filepath, model, names, all_features, device, args)
        except Exception as e:
            print(f"ERROR: {filepath}: {e}", file=sys.stderr)
            failed.append(filepath)

    # so that a calling script or a scheduler can tell
    if len(failed) > 0:
        print(f"ERROR: {len(failed)} of {len(filepaths)} inputs failed", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
