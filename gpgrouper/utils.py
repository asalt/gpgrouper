from __future__ import print_function

import os
import re
from datetime import datetime
from functools import wraps
import difflib
import textwrap

import six

import numpy as np
import pandas as pd
import click

from .containers import FASTA_FMT

end = "..."


def print_msg(*msg):
    def deco(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if six.PY3:
                print(datetime.now(), ":", *msg, end=end, flush=True)
            elif six.PY2:
                print(datetime.now(), ":", *msg, end=end)
            result = func(*args, **kwargs)
            print("done.")
            return result

        return wrapper

    return deco


# heavily inspired and borrowed
# by https://github.com/jorvis/biocode/blob/master/lib/biocode/utils.py#L149
# Joshua Orvis


def _fasta_dict_from_file(file_object, header_search="specific"):
    """
    Reads a file of FASTA entries and returns a dict for each entry.
    It parsers the headers such that `>gi|12345|ref|NP_XXXXX| Description`
    returns as `{gi: 12345, ref: 'NP_XXXXX', description : 'Description'}`
    The sequence has all whitespace removed.

    :header_search: One of `specific` or `generic`
    if `specific` tries to parse the header
    if `general` just returns each whitespace separated header
    """

    current_id = dict()
    current_seq = ""
    current_header = None
    pat = re.compile(">(\S+)\s*(.*)")
    header_pat = re.compile(r"(\w+)\|(\w+\.?\w*)?")

    def parse_header(header, pairs=True):
        keys = header_pat.findall(header)
        header_data = dict()
        for key in keys:
            if key[1] == "geneid":
                import ipdb

                ipdb.set_trace()
            header_data[key[0]] = key[1]
            # gi -> ProteinGI #, ref -> NP_XXXX
        return header_data

    for line in file_object:
        line = line.rstrip()
        m = pat.search(line)
        if m:
            ## new residue line matched, purge the existing one, if not the first
            if current_id:
                ## remove all whitespace and save
                current_seq = "".join(current_seq.split())
                current_id["sequence"] = current_seq
                yield current_id
                # current_id.clear()  # this is actually bad for list comprehensions
                # as it returns empty dictionaries

            current_seq = ""
            header = m.group(1)
            if header_search == "specific":
                current_id = parse_header(header)
            elif header_search == "generic":
                current_id = dict(header=header)
            current_id["description"] = m.group(2)

        else:
            ## python 2.6+ makes string concatenation amortized O(n)
            ##  http://stackoverflow.com/a/4435752/1368079
            current_seq += str(line)

    ## don't forget the last one
    current_seq = "".join(current_seq.split())
    current_id["sequence"] = current_seq
    yield current_id


def fasta_dict_from_file(fasta_file, header_search="specific"):
    with open(fasta_file, "r") as f:
        # yield from _fasta_dict_from_file(f, header_search=header_search)
        for v in _fasta_dict_from_file(f, header_search=header_search):
            yield v


fasta_dict_from_file.__doc__ = _fasta_dict_from_file.__doc__


def convert_tab_to_fasta(
    tabfile,
    geneid=None,
    ref=None,
    gi=None,
    homologene=None,
    taxon=None,
    description=None,
    sequence=None,
    symbol=None,
):
    HEADERS = {
        "geneid": geneid,
        "ref": ref,
        "gi": gi,
        "homologene": homologene,
        "taxon": taxon,
        "description": description,
        "sequence": sequence,
        "symbol": symbol,
    }
    CUTOFF = 0.35
    df = pd.read_table(tabfile, dtype=str)
    choices = click.Choice([x for y in [df.columns, ["SKIP"]] for x in y])
    col_names = dict()
    for h, v in HEADERS.items():
        if v is not None or v == "SKIP":
            col_names[h] = v
            continue

        closest_match = difflib.get_close_matches(h, df.columns, n=1, cutoff=CUTOFF)
        if closest_match and h != "homologene":
            col_names[h] = closest_match[0]
        else:
            print("Can not find header match for :", h)
            print("Choose from :", " ".join(choices.choices))
            choice = click.prompt(
                "Selected the correct name or SKIP",
                type=choices,
                show_default=True,
                err=True,
                default="SKIP",
            )
            col_names[h] = choice
            print()
    for _, series in df.iterrows():
        row = series.to_dict()
        gid = row.get(col_names["geneid"], "")
        ref = row.get(col_names["ref"], "")
        hid = row.get(col_names["homologene"], "")
        gi = row.get(col_names["gi"], "")
        taxon = row.get(col_names["taxon"], "")
        desc = row.get(col_names["description"], " ")
        seq = "\n".join(textwrap.wrap(row.get(col_names["sequence"], ""), width=70))
        symbol = row.get(col_names["symbol"], "")

        hid = int(hid) if hid and not np.isnan(hid) else ""
        try:
            gid = int(gid)
        except ValueError:
            pass

        r = dict(
            gid=gid,
            ref=ref,
            taxons=taxon,
            gis=gi,
            homologene=hid,
            description=desc,
            seq=seq,
            symbols=symbol,
        )
        yield (FASTA_FMT.format(**r))


def sniff_fasta(fasta):
    nrecs = 1
    fasta = fasta_dict_from_file(fasta)
    counter = 0
    REQUIRED = ("ref", "sequence")
    while counter < nrecs:
        for rec in fasta:
            if any(x not in rec for x in REQUIRED):
                raise ValueError("Invalid input FASTA")
        counter += 1
    return 0  # all good
