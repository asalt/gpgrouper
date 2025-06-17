# ===============================================================================#
import re, sys
import os
import csv
import numpy as np
from datetime import datetime
import itertools
from collections import namedtuple, defaultdict, OrderedDict, deque
import six

if six.PY3:
    from configparser import ConfigParser
elif six.PY2:
    from ConfigParser import ConfigParser
import logging
import json
from collections import defaultdict

try:
    from PIL import Image, ImageFont, ImageDraw

    # imagetitle = True
except ImportError:
    pass
# imagetitle = False

RefseqInfo = namedtuple(
    "RefseqInfo", "taxonid, geneid, homologeneid,proteingi,genefraglen"
)


def byte_formatter(b):
    conv = b / (2**10)
    if conv < 1000:
        return "{:.4f} KB".format(conv)
    elif conv > 1000:
        conv = conv / (2**10)
        if conv < 1000:
            return "{:.4f} MB".format(conv)
        elif conv < 1000:
            conv = conv / (2**10)
            return "{:.4f} GB".format(conv)


def bufcount(filename):
    """fast way to count lines in a file"""
    f = open(filename, mode="rt", encoding="latin1")
    lines = 0
    buf_size = 2048
    read_f = f.read  # loop optimization

    buf = read_f(buf_size)
    while buf:
        lines += buf.count("\n")
        buf = read_f(buf_size)

    return lines + 1  # plus 1 since I use csv_reader, which ignores the header


def rolling_window(seq, length_of_window):
    it = iter(seq)
    window = deque(maxlen=length_of_window)
    for _ in range(length_of_window):
        try:
            window.append(next(it))
        except StopIteration:
            return
    yield tuple(window)
    while it:
        try:
            _next = next(it)
            # window.append(next(it)) # this is wrong!
            window.append(_next)  # this is correct
            yield tuple(window)
        # except (StopIteration, RuntimeError):
        except StopIteration:
            return


from pyteomics import parser


def protease(seq, miscuts=2, rule="[KR]", min_pept_len=7, semi=False, **kwargs):
    nomiscuts = parser.cleave(
        seq, rule=rule, semi=semi, missed_cleavages=0, min_length=min_pept_len
    )
    all_seqs = parser.cleave(
        seq, rule=rule, semi=semi, missed_cleavages=miscuts, min_length=min_pept_len
    )

    # without M
    nomiscuts_without_m = parser.cleave(
        seq[1:], rule=rule, missed_cleavages=0, min_length=min_pept_len, semi=semi
    )
    all_seqs_without_m = parser.cleave(
        seq[1:], rule=rule, missed_cleavages=miscuts, min_length=min_pept_len, semi=semi
    )

    nomiscuts = nomiscuts | nomiscuts_without_m
    all_seqs = all_seqs | all_seqs_without_m
    return (all_seqs, len(nomiscuts))


def protease_x(
    seq,
    minlen=0,
    cutsites=tuple(),
    exceptions=None,
    miscuts=2,
    semi_tryptic=False,
    semi_tryptic_iter=6,
):
    frags = []
    chop = ""
    if exceptions is None:
        exceptions = tuple()
    import ipdb

    ipdb.set_trace()
    while seq:
        cuts = [seq.find(x) for x in cutsites]
        if len(cuts) > 0:
            if min(cuts) >= 0:
                cut = min(cuts) + 1
            else:
                cut = max(cuts) + 1
        else:
            cut = 0
        chop += seq[:cut]
        seq = seq[cut:]
        if chop == "LTEAKPAKPAPAK":
            import ipdb

            ipdb.set_trace()
        if cut == 0 or len(seq) == 0:
            if cut == 0 and len(frags) == 0:
                frags.append(chop + seq)
            elif cut == 0 and chop and seq:
                frags.append(chop + seq)
            # print(chop)
            elif len(seq) != 0:
                frags.append(seq)
            elif len(seq) == 0 and chop:
                frags.append(chop)  # special case for
            # if last amino acid is one of the cut sites
            break  # no more cuts
        if seq[0] not in exceptions:
            frags.append(chop)
            chop = ""
    merged_list = list()
    no_met = list()
    for k in range(0, miscuts):
        for ix, chunk in enumerate(rolling_window(frags, k + 2)):  # +2 to join adjacent
            merged_list.append("".join(chunk))
            if ix == 0:
                no_met = list()
                for ix, c in enumerate(chunk):
                    if ix == 0:
                        no_met.append(c[1:])
                    else:
                        no_met.append(c)
        merged_list.append("".join(no_met))

    if semi_tryptic:  #
        # MAX_ITER = 6 # let's not go beyond this, too many peptides
        MAX_ITER = semi_tryptic_iter
        # maybe have option later
        for frag in frags:
            for i in range(1, len(frag)):
                if i > MAX_ITER:
                    break
                semiN = frag[i:]
                semiC = frag[: -1 * i]
                if len(semiN) < minlen and len(semiC) < minlen:
                    break
                if len(semiN) >= minlen and semiN not in merged_list:
                    merged_list.append(semiN)
                if len(semiC) >= minlen:
                    merged_list.append(semiC)

    if len(frags) > 1:
        frags.append(frags[0][1:])  # chop off methionine
    nomiscuts_len = len([x for x in frags if len(x) >= minlen])

    return [x for x in frags + merged_list if len(x) >= minlen], nomiscuts_len


def fancyprint(ShowText, string_size=12):
    try:
        font = ImageFont.truetype("arialbd.ttf", string_size)  # load the font
        font_import = True
    except IOError:
        font_import = False
        print(ShowText)
    if font_import:
        size = font.getsize(ShowText)  # calc the size of text in pixels
        image = Image.new("1", size, 1)  # create a b/w image
        draw = ImageDraw.Draw(image)
        draw.text((0, 0), ShowText, font=font)  # render the text to the bitmap
        for rownum in range(size[1]):
            # scan the bitmap:
            # print ' ' for black pixel and
            # print '#' for white one
            line = []
            for colnum in range(size[0]):
                if image.getpixel((colnum, rownum)):
                    line.append(" "),
                else:
                    line.append("#"),
            print("".join(line))


label_pos = re.compile(
    """
    (?<!Label\:)  # not preceded by
    (\d+|N-Term)      # Match numbers,
    (?!       # only if it's not followed by..."
     [^(]*    #   any number of characters except opening parens" +
     \\)      #   followed by a closing parens" +
    )         # End of lookahead""",
    re.VERBOSE,
)

inside_paren = re.compile(r"\((\S+)\)")
amino_acids = list("ACDEFGHIKLMNPQRSTVWY")


def seq_modi(sequence, modifications, no_count=None):
    """
    function to output a modified string of sequence that includes all modifications at the appropriate place.
    Modified amino acids are indicated in the input 'sequence' by a lowercase letter.
    A dictionary of potential modification keys and their corresponding substitutions in the primary sequence is provided;
    all other modifications should be ignored.
    Sometimes N-terminal modifications are present, this should be considered the same as a modification at the first amino acid.

    modifications is a 1-based index of amino acid and modification delimited by semicolon (; ):
    M1(Oxidation) is first amino acid methionine Oxidized

    """
    if no_count is None:
        no_count = tuple()
    # TODO : have separate collections for labeled and non-labeled modifications
    seqmodi = ""
    seqlist = list(sequence)
    modi_len = 0  # default length is zero, can change
    label = 0  # default label = 0, can change
    label_search = re.search(r"(Label\S+)(?=\))", modifications)  # for SILAC
    if label_search:
        label = label_search.group(0)

    # if not any(c for c in sequence if (c.islower() or c == 'X')): # check if any modifications to deal with
    if (
        not any(c == "X" for c in sequence) and not modifications
    ):  # check if any modifications to deal with
        return sequence, sequence, 0, label
    modkeys = inside_paren.findall(modifications)
    modi_len = len([x for x in modkeys if not any(y in x for y in no_count)])
    # any(mod in x for mod in to_count)])
    # modkeys = re.findall(r'(\([^\)]+\))',modifications)  #get all modifications
    # modpos = re.findall(r'[A-Z]([0-9]+)',modifications)  # get all modification positions
    modpos = label_pos.findall(modifications)
    modpos = [
        0 if x.lower() == "n-term" else int(x) - 1 for x in modpos
    ]  # n terminal is at first position
    mod_dict = defaultdict(list)
    for (key, value) in zip(modpos, modkeys):
        mod_dict[key].append(value.lower())
    for key, values in mod_dict.items():
        mod_dict[key] = sorted(values)
    for ix, s in enumerate(sequence.upper()):
        # print(ix, seqmodi)
        if ix in mod_dict:
            if s == "X":  # deal with the X first
                to_replace = [x for x in mod_dict[ix] if x in amino_acids]
                if len(to_replace) == 1:
                    if seqlist[ix].islower():
                        replaced_aa = to_replace[0].lower()
                    else:
                        replaced_aa = to_replace[0]
                    seqmodi += replaced_aa.upper()  # all seqmodi AAs are upper case
                    seqlist[ix] = replaced_aa
                    mod_dict[ix].remove(replaced_aa.upper())
                elif len(to_replace) == 0:
                    pass  # not an amino acid (listed above at least)
                    # Probably an unidentified mass of the form X10(110.0)
                else:
                    print(
                        "Error parsing sequence {} with modifications {}".format(
                            sequence, modifications
                        )
                    )
                    seqmodi += s
            else:
                seqmodi += s
        else:
            seqmodi += s
        if ix not in mod_dict:
            continue
        for modi in mod_dict[ix]:
            if sequence[ix].islower() and modi not in amino_acids:
                modi_ = modi.lower()
            else:
                modi_ = modi
            # seqmodi += modi
            seqmodi += "({})".format(modi_)
    sequence = "".join(seqlist)
    return sequence, seqmodi, modi_len, label
    # return sequence, seqmodi, modi_len


def _seq_modi_old(sequence, modifications, keeplog=True):

    """
    function to output a modified string of sequence that includes all
    modifications at the appropriate place.
    Modified amino acids are indicated in the input 'sequence' by a lowercase
    letter.
    A dictionary of potential modification keys and their corresponding
    substitutions in the primary sequence is provided;
    all other modifications should be ignored.
    Sometimes N-terminal modifications are present, this should be considered
    the same as a modification at the first amino acid.

    also modifies sequence to fill in Xs with the predicted modification
    example : AAAx(P)R --> AAAPR
    """
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    amino_acids = ["(" + x + ")" for x in amino_acids]
    modtext = {
        "(DeStreak)": "(des)",
        "(Deamidated)": "(dam)",
        "(Carbamidomethyl)": "(car)",
        "(Oxidation)": "(oxi)",
        "(Phospho)": "(pho)",
        "(Prot)(Acetyl)": "(ace)",
        "(Acetyl)": "(ace)",
        "(GlyGly)": "(gg)",
        "(Label:13C(6))": "(lab)",
        "(Label:13C(6)+GlyGly)": "(labgg)",
    }
    seqmodi = ""
    seqlist = list(sequence)
    modi_len = 0  # default length is zero, can change
    label = 0  # default label = 0, can change
    if any(c for c in sequence if (c.islower() or c == "X")):
        # check if any modifications to deal with
        if "Label" in modifications:
            # routine for grabbing everything enclosed in parenthesis, including
            # nested. Easier to write than regex when dealing with nested
            label = 1  # for indication of label on peptide
            openb, modi, modkeys = 0, "", []
            for e in modifications:
                if e == "(":
                    openb += 1
                elif e == ")":
                    openb += -1
                if openb:
                    modi += e
                elif not openb:
                    if modi:
                        modi += e  # add the last parenthesis
                        modkeys.append(modi)
                    modi = ""
        else:
            modkeys = re.findall(r"(\([^\)]+\))", modifications)
            # get all modifications

        modpos = re.findall(r"[A-Z]([0-9]+)", modifications)
        # get all modification positions
        modpos = [int(d) - 1 for d in modpos]  # convert all to integers
        if "N-Term" in modifications:
            modpos.insert(0, 0)
            # first of the modkeys will be the N-Term modification
        modi_len = len(modpos)
        mod_dict = defaultdict(list)
        for (key, value) in zip(modpos, modkeys):
            mod_dict[key].append(value)
        for key in mod_dict:
            mod_dict[key] = sorted(mod_dict[key])

        for ix, s in enumerate(sequence.upper()):
            seqmodi += s
            # print(ix, seqmodi)
            if ix in mod_dict:
                if s == "X":  # deal with the X first
                    to_replace = [x for x in mod_dict[ix] if x in amino_acids]

                    if len(to_replace) == 1:
                        # print(seqlist[ix],to_replace)
                        if seqlist[ix].islower():
                            seqlist[ix] = to_replace[0][1].lower()
                        else:
                            seqlist[ix] = to_replace[0][1]
                        modi_len += -1  # don't count X == amino acid as a
                        # modification since technically it is not a PTM
                        # mod_dict[ix].remove(to_replace)  # not sure if we want
                        # to remove this or not, but we certainly can if we want
                    elif len(to_replace) == 0:
                        pass  # not an amino acid (listed above at least)
                    # Probably an unidentified mass of the form X10(110.0)

                    else:
                        print(
                            "Error parsing sequence {} with "
                            "modifications {}".format(sequence, modifications)
                        )
                for modi in mod_dict[ix]:
                    if modi in modtext:
                        seqmodi += modtext[modi]
                    elif modi in amino_acids:
                        seqmodi += modi  # for the X amino acids
                    else:
                        if keeplog:
                            logging.warning(
                                "New modification {} that"
                                " is not found in sequence {}".format(modi, sequence)
                            )

    sequence = "".join(seqlist)
    if not seqmodi:
        seqmodi = sequence
    return sequence, seqmodi, modi_len, label


def _count_modis_maxquant(modi, labeltype):
    if modi == "Unmodified":
        return 0
    count = modi.count(",") + 1  # multiple modis are separated by a comma
    if labeltype == "TMT":
        count -= modi.lower().count("tmt")
    elif labeltype == "iTRAQ":
        count -= modi.lower().count("itraq")
    return count


def count_modis_maxquant(df, labeltype):
    return df.apply(
        lambda x: _count_modis_maxquant(x["Modifications"], labeltype), axis=1
    )


def count_modis_seqmodi(df, labeltype):
    if labeltype == "TMT":
        return df.SequenceModi.str.count(r"[\(|\[]") - df.SequenceModi.str.count(
            r"[\(|\[][TMT|tmt|305|229]"
        ).fillna(0)
    elif labeltype == "iTRAQ":
        return df.SequenceModi.str.count(r"\(") - df.SequenceModi.str.count(
            r"[\(|\[][iTRAQ|itraq]"
        ).fillna(0)
    else:
        return df.SequenceModi.str.count(r"[\(|\[]").fillna(0)


def calculate_miscuts(seq, targets=None, exceptions=None):
    """Calculates number of miscuts for a given sequence
    using amino acids given in targets"""
    if exceptions is None:
        exceptions = tuple()
    not_miscut = ["".join(x) for x in itertools.product(targets, exceptions)]
    miscuts = sum(seq.count(x) for x in targets)
    miscuts_not = sum(seq.count(x) for x in not_miscut)
    miscuts -= miscuts_not
    if not any(seq[-1] == x for x in targets):  # then at the C terminal
        return miscuts
    return miscuts - 1


import hashlib


def md5sum(filename, blocksize=65536):
    hash = hashlib.md5()
    with open(filename, "rb") as f:
        for block in iter(lambda: f.read(blocksize), b""):
            hash.update(block)
    return hash.hexdigest()


def write_md5(filename, chsum):
    with open(filename, "w") as f:
        f.write(chsum)


# def _fasta_dict_from_file(file_object, header_search="specific"): # this one is old
#     """
#     Reads a file of FASTA entries and returns a dict for each entry.
#     It parsers the headers such that `>gi|12345|ref|NP_XXXXX| Description`
#     returns as `{gi: 12345, ref: 'NP_XXXXX', description : 'Description'}`
#     The sequence has all whitespace removed.
#
#     :header_search: One of `specific` or `generic`
#     if `specific` tries to parse the header
#     if `general` just returns each whitespace separated header
#     """
#
#     current_id = dict()
#     current_seq = ""
#     current_header = None
#     pat = re.compile(">(\S+)\s*(.*)")
#     # header_pat = re.compile(r'(\w+)\|(\w+\.?\w*)?')
#     header_pat = re.compile(r"(\w+)\|(\S+)?")
#     headers = ["gi", "ref", "geneid", "homologene", "taxon", "symbol"]
#     header_pat = re.compile(
#         r"gi\|(\S*)\|ref\|(\S*)\|geneid\|(\S*)\|homologene\|(\S*)\|taxon\|(\S*)\|symbol\|(\S*)"
#     )
#
#     def parse_header(header, pairs=True):
#         header_dict = dict()
#         for h in headers:
#             try:
#                 head_value = re.search(h + "\|(\w+\.?\d?)\|", header).group(1)
#             except AttributeError:
#                 head_value = "" if h != "geneid" else header
#             header_dict[h] = head_value
#
#         # try:
#         #     keys = header_pat.findall(header)[0]
#         # except IndexError:
#         #     # warnings.warn('Error parsing {}, using full as gene id'.format(header))
#         #     return dict(gi='', ref='', geneid=header, homologene='', taxon='', symbol='')
#         # header_data = dict()
#         # for h, key in zip(headers, keys):
#         #     header_data[h] = key
#         # for key in keys:
#         #     header_data[key[0]] = key[1]
#         # gi -> ProteinGI #, ref -> NP_XXXX
#         return header_dict
#
#     for line in file_object:
#         line = line.rstrip()
#         m = pat.search(line)
#         if m:
#             ## new residue line matched, purge the existing one, if not the first
#             if current_id:
#                 ## remove all whitespace and save
#                 current_seq = "".join(current_seq.split())
#                 current_id["sequence"] = current_seq
#                 yield current_id
#                 # current_id.clear()  # this is actually bad for list comprehensions
#                 # as it returns empty dictionaries
#
#             current_seq = ""
#             header = m.group(1)
#             if header_search == "specific":
#                 current_id = parse_header(header)
#             elif header_search == "generic":
#                 current_id = dict(header=header)
#             current_id["description"] = m.group(2)
#
#         else:
#             ## python 2.6+ makes string concatenation amortized O(n)
#             ##  http://stackoverflow.com/a/4435752/1368079
#             current_seq += str(line)
#
#     ## don't forget the last one
#     current_seq = "".join(current_seq.split())
#     current_id["sequence"] = current_seq
#     yield current_id
#
#
# def fasta_dict_from_file(fasta_file, header_search="specific"):
#     with open(fasta_file, "r") as f:
#         # yield from _fasta_dict_from_file(f, header_search=header_search)
#         for v in _fasta_dict_from_file(f, header_search=header_search):
#             yield v

# fasta_dict_from_file.__doc__ = _fasta_dict_from_file.__doc__

from RefProtDB.utils import fasta_dict_from_file, _fasta_dict_from_file


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
