from __future__ import print_function

import os
import re
import logging
from datetime import datetime
from functools import wraps
import difflib
import textwrap

import six

import numpy as np
import pandas as pd
import click

from gpgrouper.mapper import fetch_info_from_db


logger = logging.getLogger(__name__)

end = "..."

header_pat = re.compile(r"(\w+)\|([\w-]+)\|([\w\-]+)?")

FASTA_FMT = """>gi|{gis}|ref|{ref}|geneid|{gid}|homologene|{homologene}|taxon|{taxons}|symbol|{symbols} {description}\n{seq}"""


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


def fetch_entrez_from_uniprot(uniprot_id, fetch_online=True):
    """Fetch Entrez Gene ID given a UniProt ID using your existing lookup function."""
    info = fetch_info_from_db(
        [uniprot_id], key_type="uniprot", fetch_online=fetch_online
    )
    return info.get(uniprot_id, {}).get("entrezgene", None)


def fetch_symbol_from_uniprot(uniprot_id, fetch_online=True):
    """Fetch Entrez Gene ID given a UniProt ID using your existing lookup function."""
    info = fetch_info_from_db(
        [uniprot_id], key_type="uniprot", fetch_online=fetch_online
    )
    return info.get(uniprot_id, {}).get("symbol", None)
    # return "?"


header_pat = re.compile(r"(\w+)\|([\w-]+)")
# desc_pat = re.compile(r"(OS|OX|GN|PE|SV)=([\w\s.-]+)")
desc_pat = re.compile(r"(OS|OX|GN|PE|SV)=([\w\s.-]+?)(?=\s(?:OS|OX|GN|PE|SV)=|\s*$)")
# this works ! - to make it not greedy.
# maybe could be written more easily/programatically  in the future


def further_parse_description(description):
    """
    Extracts metadata (OS, OX, GN, PE, SV) from the description field.
    Returns a dictionary with parsed values.
    """
    metadata = {}

    for key, value in desc_pat.findall(description):
        key_map = {
            "OX": "taxonid",
            "GN": "gene_name",
            "OS": "organism",
            "PE": "protein_evidence",
            "SV": "sequence_version",
        }
        metadata[key_map[key]] = value.strip()
    os_idx = description.find("OS=")
    if os_idx == -1:
        os_idx = len(description)

    metadata["description"] = description[:os_idx].strip()

    return metadata


def parse_header(header, fetch_online=True):
    """
    Parses a FASTA header and extracts ALL key-value pairs.
    Supports UniProt, NCBI, and other common formats.
    Captures the remaining description text at the end.
    """
    TOKEEP = {
        "gi",
        "ref",
        "geneid",
        "symbol",
        "taxon",
        "sp",
        "spid",
        "spacc",
        "spref",
        "ENSG",
        "ENST",
        "ENSP",
    }

    # Find all key-value pairs
    matches = header_pat.findall(header)

    header_data = {"raw_header": header}
    uniprot_id = None

    last_match_end = 0  # To track where the last regex match ended

    for key, value in matches:
        last_match_end = header.find(value, last_match_end) + len(
            value
        )  # Update last match position

        if key.startswith("rev"):
            continue  # Ignore reverse/decoy entries

        if key in TOKEEP:
            header_data[key] = value
            if key == "sp":  # UniProt ID case
                uniprot_id = value

    # Capture the remaining description text
    description = header[last_match_end:].strip()
    if description:

        header_data["description"] = description.lstrip("|")
        more_description = further_parse_description(description)

        if (
            "taxonid" in more_description
            and "taxon" not in header_data
            or header_data.get("taxon") == ""
        ):
            header_data["taxon"] = more_description.get("taxonid")
        if (
            "gene_name" in more_description
            and "gene_name" not in header_data
            or header_data.get("gene_name") == ""
        ):
            header_data["symbol"] = more_description.get("gene_name")
            # is this working or not?
        header_data["description"] = more_description.get(
            "description", header_data["description"]
        )

        # if (
        #     "gene_name" in more_description
        #     and "description" not in header_data
        #     or header_data.get("description") == ""
        # ):
        #     header_data["description"] = more_description["gene_name"]
        # if not header_data["raw_header"].startswith("contam"):
        #     import ipdb

        # ipdb.set_trace()
        # header_data.update(
        #     further_parse_description(description)
        # )  # Merge parsed description fields

    # Auto-fetch Entrez ID if only UniProt ID is present

    if uniprot_id and "geneid" not in header_data:
        entrez_id = fetch_entrez_from_uniprot(uniprot_id, fetch_online=fetch_online)
        if entrez_id:
            header_data["geneid"] = entrez_id

    if uniprot_id and "symbol" not in header_data:
        entrez_id = fetch_symbol_from_uniprot(uniprot_id, fetch_online=fetch_online)
        if entrez_id:
            header_data["symbol"] = entrez_id

    # print(header_data)
    return header_data


# heavily inspired and borrowed
# by https://github.com/jorvis/biocode/blob/master/lib/biocode/utils.py#L149
# Joshua Orvis


def _fasta_dict_from_file(file_object, header_search="specific", fetch_online=True):
    """
    Reads a file of FASTA entries and returns a dict for each entry.
    Parses headers for various IDs and associates them with sequences.
    """

    current_id = dict()
    current_seq = ""
    pat = re.compile(r">(\S+)\s*(.*)")

    for line in file_object:
        line = line.rstrip()
        m = pat.search(line)
        if m:
            # Save previous entry
            if current_id:
                current_seq = "".join(current_seq.split())
                current_id["sequence"] = current_seq
                yield current_id

            current_seq = ""
            header = m.group(1)
            current_id = (
                parse_header(line.lstrip(">"), fetch_online=fetch_online)
                if header_search == "specific"
                else {"raw_header": header}
            )
            current_id["description"] = m.group(2)
        else:
            current_seq += str(line)

    # Save last entry
    current_seq = "".join(current_seq.split())
    current_id["sequence"] = current_seq
    if (
        current_id.get("geneid") is None
        or pd.isna(current_id.get("geneid"))
        # or current_id.get("geneid") == ""
        #     current_id["ENSP"] == "Cont_P62937"
    ):

        geneid_value = current_id.get(
            "ENSG",
            current_id.get("ENSP", current_id.get("sp", current_id.get("raw_header"))),
        )
        current_id["geneid"] = geneid_value
    yield current_id


def fasta_dict_from_file(fasta_file, header_search="specific", fetch_online=True):
    with open(fasta_file, "r") as f:
        # yield from _fasta_dict_from_file(f, header_search=header_search)
        for v in _fasta_dict_from_file(
            f, header_search=header_search, fetch_online=fetch_online
        ):
            yield v


def load_fasta_into_dataframe(fasta_file, header_search="specific", fetch_online=True):
    res = pd.DataFrame(
        fasta_dict_from_file(
            fasta_file, header_search=header_search, fetch_online=False
        )
    )
    if not fetch_online:
        return res
    if "geneid" in res.columns and res["geneid"].isna().any():
        subsel = res[res["geneid"].isna()]
        if "sp" in subsel.columns:
            to_query = subsel["sp"].unique()
            uni_gene_mapping = fetch_info_from_db(to_query, key_type="uniprot")
            for k, v in uni_gene_mapping.items():
                res.loc[res["sp"] == k, "geneid"] = v.get("entrezgene", k)
    return res


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
