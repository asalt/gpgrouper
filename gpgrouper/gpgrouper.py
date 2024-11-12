# PyGrouper - Alex Saltzman
# this file does too many things, difficult to debug
from __future__ import print_function
import tqdm

import re, os, sys, time
import itertools
import json
import logging
from time import sleep
from collections import defaultdict
from functools import partial
from math import ceil
from warnings import warn
import warnings
import six

if six.PY3:
    from configparser import ConfigParser
elif six.PY2:
    from ConfigParser import ConfigParser
from itertools import repeat
import traceback
import multiprocessing
from copy import deepcopy as copy
from functools import lru_cache

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from RefProtDB.utils import fasta_dict_from_file
from pyteomics import parser

parser.expasy_rules["trypsin/P"] = "[KR]"
parser.expasy_rules["chymotrypsin"] = parser.expasy_rules[
    "chymotrypsin high specificity"
]

from . import _version
from .subfuncts import *

# from ._orig_code import timed
pd.set_option(
    "display.width",
    170,
    "display.max_columns",
    500,
)
__author__ = "Alexander B. Saltzman"
__copyright__ = _version.__copyright__
__credits__ = ["Alexander B. Saltzman", "Anna Malovannaya"]
__license__ = "BSD 3-Clause"
__version__ = _version.__version__
__maintainer__ = "Alexander B. Saltzman"
__email__ = "saltzman@bcm.edu"
program_title = "gpGrouper v{}".format(__version__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

logfilename = program_title.replace(" ", "_") + ".log"
logging.basicConfig(
    format=("[{levelname}] {asctime}s {message}s {message}"),
    level=logging.INFO,
    style="{",
    # level=verbosity_dict[config.verbosity],
)


logging.info("{}: Initiating {}".format(datetime.now(), program_title))

SEP = ";"

labelflag = {
    "none": 0,  # hard coded number IDs for labels
    "TMT_126": 1260,
    "TMT_127_C": 1270,
    "TMT_127_N": 1271,
    "TMT_128_C": 1280,
    "TMT_128_N": 1281,
    "TMT_129_C": 1290,
    "TMT_129_N": 1291,
    "TMT_130_C": 1300,
    "TMT_130_N": 1301,
    "TMT_131_N": 1310,
    "TMT_131_C": 1311,
    "TMT_132_N": 1321,
    "TMT_132_C": 1320,
    "TMT_133_N": 1331,
    "TMT_133_C": 1330,
    "TMT_134_N": 1340,
    "TMT_134_C": 1341,
    "TMT_135":   1350,
    "iTRAQ_114": 113,
    "iTRAQ_114": 114,
    "iTRAQ_115": 115,
    "iTRAQ_116": 116,
    "iTRAQ_117": 117,
    "iTRAQ_118": 118,
    "iTRAQ_119": 119,
    "iTRAQ_121": 121,
}
flaglabel = {v: k for k, v in labelflag.items()}

E2G_COLS = [
    "EXPRecNo",
    "EXPRunNo",
    "EXPSearchNo",
    "EXPLabelFLAG",
    "AddedBy",
    # 'CreationTS', 'ModificationTS',
    "GeneID",
    "GeneSymbol",
    "Description",
    "TaxonID",
    "HIDs",
    "PeptidePrint",
    "GPGroup",
    "GPGroups_All",
    "ProteinGIs",
    "ProteinRefs",
    "ProteinGI_GIDGroups",
    "ProteinGI_GIDGroupCount",
    "ProteinRef_GIDGroups",
    "ProteinRef_GIDGroupCount",
    "IDSet",
    "IDGroup",
    "IDGroup_u2g",
    "SRA",
    "Coverage",
    "Coverage_u2g",
    "PSMs",
    "PSMs_u2g",
    "PeptideCount",
    "PeptideCount_u2g",
    "PeptideCount_S",
    "PeptideCount_S_u2g",
    "AreaSum_u2g_0",
    "AreaSum_u2g_all",
    "AreaSum_max",
    "AreaSum_dstrAdj",
    "GeneCapacity",
    "iBAQ_dstrAdj",
]

DATA_ID_COLS = [
    "EXPRecNo",
    "EXPRunNo",
    "EXPSearchNo",
    "Sequence",
    "PSMAmbiguity",
    "Modifications",
    "ActivationType",
    "DeltaScore",
    "DeltaCn",
    "Rank",
    "SearchEngineRank",
    "PrecursorArea",
    "q_value",
    "PEP",
    "IonScore",
    "MissedCleavages",
    "IsolationInterference",
    "IonInjectTime",
    "Charge",
    "mzDa",
    "MHDa",
    "DeltaMassDa",
    "DeltaMassPPM",
    "RTmin",
    "FirstScan",
    "LastScan",
    "MSOrder",
    "MatchedIons",
    "SpectrumFile",
    "AddedBy",
    "oriFLAG",
    # 'CreationTS', 'ModificationTS',
    # "GeneID",
    "GeneIDs_All",
    "GeneIDCount_All",
    "ProteinGIs",
    "ProteinGIs_All",
    "ProteinGICount_All",
    "ProteinRefs",
    "ProteinRefs_All",
    "ProteinRefCount_All",
    "HIDs",
    "HIDCount_All",
    "TaxonID",
    "TaxonIDs_All",
    "TaxonIDCount_All",
    "PSM_IDG",
    "SequenceModi",
    "SequenceModiCount",
    "LabelFLAG",
    "AUC_UseFLAG",
    "PSM_UseFLAG",
    "Peak_UseFLAG",
    "SequenceArea",
    "PeptRank",
]


DATA_QUANT_COLS = [
    "EXPRecNo",
    "EXPRunNo",
    "EXPSearchNo",
    "LabelFLAG",
    "FirstScan",
    "SpectrumFile",
    "Charge",
    "GeneID",
    "SequenceModi",
    "PeptRank",
    "ReporterIntensity",
    "PrecursorArea",
    "PrecursorArea_split",
    "SequenceArea",
    "PrecursorArea_dstrAdj",
]

GENE_QUAL_COLS = [
    "EXPRecNo",
    "EXPRunNo",
    "EXPSearchNo",
    "GeneID",
    "LabelFLAG",
    "ProteinRef_GIDGroupCount",
    "TaxonID",
    "SRA",
    "GPGroups_All",
    "IDGroup",
    "IDGroup_u2g",
    "ProteinGI_GIDGroupCount",
    "HIDs",
    "PeptideCount",
    "IDSet",
    "Coverage_u2g",
    "Symbol",
    "Coverage",
    "PSMs_S_u2g",
    "ProteinGIs",
    "Description",
    "PSMs",
    "PeptideCount_S",
    "ProteinRefs",
    "PSMs_S",
    "HomologeneID",
    "PeptideCount_u2g",
    "GeneSymbol",
    "GPGroup",
    "PeptideCount_S_u2g",
    "PeptidePrint",
    "PSMs_u2g",
    "GeneCapacity",
    "ProteinGI_GIDGroups",
    "ProteinRef_GIDGroups",
]

GENE_QUANT_COLS = [
    "EXPRecNo",
    "EXPRunNo",
    "EXPSearchNo",
    "LabelFLAG",
    "GeneID",
    "SRA",
    "AreaSum_u2g_0",
    "AreaSum_u2g_all",
    "AreaSum_max",
    "AreaSum_dstrAdj",
    "iBAQ_dstrAdj",
]


# only SequenceArea and PrecursorArea_dstrAdj?

_EXTRA_COLS = [
    "LastScan",
    "MSOrder",
    "MatchedIons",
]  # these columns are not required to be in the output data columns

try:
    from PIL import Image, ImageFont, ImageDraw

    imagetitle = True
except ImportError:
    imagetitle = False

if six.PY2:

    class DirEntry:
        def __init__(self, f):
            self.f = f

        def is_file(self):
            return os.path.isfile(self.f)

        def is_dir(self):
            return os.path.isdir(self.f)

        @property
        def name(self):
            return self.f

    def scandir(path="."):
        files = os.listdir(".")
        for f in files:
            yield DirEntry(f)

    os.scandir = scandir


def _apply_df(input_args):
    df, func, i, func_args, kwargs = input_args
    return i, df.apply(func, args=(func_args), **kwargs)


def apply_by_multiprocessing(df, func, workers=1, func_args=None, **kwargs):
    """
    Spawns multiple processes if has os.fork and workers > 1
    """
    if func_args is None:
        func_args = tuple()

    if workers == 1 or not hasattr(os, "fork"):
        result = _apply_df(
            (
                df,
                func,
                0,
                func_args,
                kwargs,
            )
        )
        return result[1]

    workers = min(workers, len(df))  # edge case where df has less rows than workers
    workers = max(workers, 1)  # has to be at least 1

    # pool = multiprocessing.Pool(processes=workers)
    with multiprocessing.Pool(processes=workers) as pool:
        result = pool.map(
            _apply_df,
            [
                (
                    d,
                    func,
                    i,
                    func_args,
                    kwargs,
                )
                for i, d in enumerate(np.array_split(df, workers))
            ],
        )
        # pool.close()

    result = sorted(result, key=lambda x: x[0])

    return pd.concat([x[1] for x in result])


def quick_save(df, name="df_snapshot.p", path=None, q=False):
    import pickle

    # import RefSeqInfo

    if path:
        name = path + name
    # df.to_csv('test_matched.tab', index=False, sep='\t')
    pickle.dump(df, open(name, "wb"))
    print("Pickling...")
    if q:
        print("Exiting prematurely")
        sys.exit(0)


# @lru_cache(maxsize=None)
# def _map_to_gene(proteins, database, column='ProteinList', sep=';', regex=None):

#     protein_ids = list()
#     # for protein in row[column].split(sep):
#     for protein in proteins.split(sep):
#         protein_ids.append( regex.search(protein).group() )

#     matches = database[ database[identifier].isin(protein_ids) ]

#     return '|'.join(str(x) for x in matches.index)


def map_to_gene(usrdata, column, identifier="ref"):
    """
    map protein gi / ref accession directly to gene, bypassing fasta file searching
    """
    if column not in usrdata.df:
        err = "`{}` not in input.".format(column)
        usrdata.EXIT_CODE = 0
        usrdata.ERROR = "`{}` not in input.".format(column)
        return ""

    # try:
    #     import mygene
    # except ImportError:
    #     warn('MyGeneInfo not installed, cannot connect to external database')
    #     geneinfo = None

    class QueryExternal:
        try:
            import mygene

            geneinfo = mygene.MyGeneInfo()
        except ImportError:
            warn("MyGeneInfo not installed, cannot connect to external database")
            geneinfo = None

        def __init__(self):
            self.saved = dict()

        def __call__(self, queries, scopes="refseq,accession"):
            if self.geneinfo is None:
                return

            subqueries = set(queries) - set(self.saved.keys())
            if subqueries:
                results = self.geneinfo.querymany(
                    subqueries, scopes=scopes, as_dataframe=True
                )
            else:
                results = pd.DataFrame()
            # results = results.drop_duplicates(on='entrezgene')

            # save results
            for ix, row in results.iterrows():
                self.saved[ix] = row.to_dict()

            saved_queries = set(queries) - set(subqueries)
            newrows = list()
            for s in saved_queries:
                newrows.append(self.saved[s])

            results = pd.concat([results, pd.DataFrame(newrows, index=saved_queries)])

            return results

    queryexternal = QueryExternal()

    if identifier not in ("ref", "gi"):
        raise ValueError("must be either `ref` or `gi`. Uniprot support coming...")

    from glob import glob

    MAPPING = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "mappings",
        "human_mouse_mapping_2019May.tsv",
    )

    OTHER_MAPPINGS = glob(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "mappings", "./mapping*.tsv"
        )
    )

    # MAPPING = os.path.join(os.path.dirname(os.path.realpath(__file__)),
    #                        './human_mouse_mapping_2019Jan.tsv')
    global database
    database = pd.read_table(MAPPING, dtype=str)
    database["capacity"] = database.capacity.astype(float)

    # load other mappings..
    if OTHER_MAPPINGS:
        _frames = list()
        for m in OTHER_MAPPINGS:
            _frames.append(pd.read_table(m))
        database = pd.concat([database, *_frames]).reset_index(drop=True)

    MAX_DATABASE_INDEX = database.index.max()

    if identifier == "ref":
        pat = r"(?<=ref\|)(.*)(?=\|)"
    elif identifier == "gi":
        pat = r"(?<=gi\|)(.*)(?=\|)"
    regex = re.compile(pat)

    @lru_cache(maxsize=None)
    def _map_to_gene(proteins, sep=";", regex=None):
        global database  # because cannot hash it

        protein_ids = list()
        # for protein in row[column].split(sep):
        for protein in proteins.split(sep):
            reg_res = regex.search(protein)
            if reg_res is None:
                reg_res = protein.strip()  # try assuming it is just identifiers
            else:
                reg_res = reg_res.group()
            protein_ids.append(reg_res)

        matches = database[database[identifier].isin(protein_ids)]
        nomatches = [x for x in set(protein_ids) - set(matches[identifier]) if x]

        # if nomatches:
        if nomatches and identifier != "gi":  # GI doesn't work
            nomatch_res = queryexternal(nomatches)

            todrop = ["_score", "_id", "name", "notfound"]
            todrop = [x for x in todrop if x in nomatch_res]
            if "notfound" in nomatch_res:  # this column is added for unfound queries
                ixs = nomatch_res[nomatch_res["notfound"] != True].index
            else:
                ixs = nomatch_res.index

            # if not nomatch_res.empty:
            nomatch_res = (
                nomatch_res.loc[ixs]
                .drop(todrop, axis=1)
                .reset_index()
                .rename(
                    columns={
                        "entrezgene": "geneid",
                        "taxid": "taxon",
                        "query": identifier,
                    }
                )
            )
            nomatch_res.index = np.arange(
                database.index.max() + 1, database.index.max() + len(nomatch_res) + 1
            )

            if not nomatch_res.empty:
                # not very efficient...
                database = database.append(nomatch_res)

            # now get the indices again...
            matches = database[database[identifier].isin(protein_ids)]

        return "|".join(str(x) for x in matches.index)

        # geneids = SEP.join(str(x) for x in matches.geneid.unique())
        # return geneids

    ## cannot hash the function when passing database dataframe as argument
    # info = usrdata.df[column].pipe(apply_by_multiprocessing,
    #                                _map_to_gene,
    #                                func_args=(database,),
    #                                sep=';',
    #                                regex=regex,
    #                                workers=WORKERS,
    #                                axis=1
    # )
    # usrdata.df['metadatainfo'] = info

    metadatainfo = (
        usrdata.df[column].fillna("").apply(_map_to_gene, sep=";", regex=regex)
    )
    # metadatainfo, nonmatch_res = list(zip(*res))
    usrdata.df["metadatainfo"] = metadatainfo

    added_entries = database.loc[MAX_DATABASE_INDEX + 1 :]

    if not added_entries.empty:
        NEW_DB = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "mappings",
            "mapping_{}.tsv".format(datetime.now().toordinal()),
        )

        added_entries.to_csv(NEW_DB, index=False, sep="\t")
        print("Wrote new mapping file with {} entries".format(added_entries.pipe(len)))

    # usrdata.df['metadatainfo'], nomatch_res = list(zip(usrdata.df.head()[column].apply(_map_to_gene, sep=';', regex=regex))

    # usrdata.df['metadatainfo'], nomatch_res = usrdata.df[column].apply(_map_to_gene, sep=';', regex=regex)

    database["taxon"] = (
        database["taxon"].fillna(-1).apply(int).apply(str).replace("-1", "")
    )

    extract_peptideinfo(usrdata, database)
    # usrdata.df['miscuts_not']
    # usrdata.df['MissedCleavages'] = -1

    usrdata.to_log("Mapping proteins to genes directly, not using database search.")

    return database


def _get_rawfile_info(path, spectraf):
    if path is None:
        path = "."
    if not os.path.isdir(path):
        return ("not found, check rawfile path", "not found")
    for f in os.listdir(path):
        if f == spectraf:
            rawfile = os.path.abspath(os.path.join(path, f))
            break
    else:
        return ("not found", "not found")

    fstats = os.stat(rawfile)
    mod_date = datetime.fromtimestamp(fstats.st_mtime).strftime("%m/%d/%Y %H:%M:%S")
    size = byte_formatter(fstats.st_size)
    return (size, mod_date)


def _spectra_summary(spectraf, data):
    """Calculates metadata per spectra file.
    The return order is as follows:

    -minimum RT_min
    -maximum RT_min
    -min IonScore
    -max IonScore
    -min q_value
    -max q_value
    -min PEP
    -max PEP
    -min Area (precursor, exculding zeros)
    -max Area
    -PSM Count
    -median DeltaMassPPM
    """
    data = data[data.SpectrumFile == spectraf]

    RT_min = data.RTmin.min()
    RT_max = data.RTmin.max()
    IonScore_min = data.IonScore.min()
    IonScore_max = data.IonScore.max()
    q_min = data.q_value.min()
    q_max = data.q_value.max()
    PEP_min = data.PEP.min()
    PEP_max = data.PEP.max()
    area_min = data[data.PrecursorArea != 0].PrecursorArea.min()
    area_max = data.PrecursorArea.max()
    count = len(data[data.PSM_UseFLAG == 1])
    dmass_median = data.DeltaMassPPM.median()
    return (
        RT_min,
        RT_max,
        IonScore_min,
        IonScore_max,
        q_min,
        q_max,
        PEP_min,
        PEP_max,
        area_min,
        area_max,
        count,
        dmass_median,
    )


def spectra_summary(usrdata, psm_data):
    """Summaries the spectral files included in an analysis.

    Args:
        usrdata: a UserData instance with the data loaded


    Returns:
        A pandas DataFrame with relevant columns, ready for export

        if the raw files cannot be found at usrdata.rawfiledir,
        then 'not found' is returned for those columns
    """
    msfdata = pd.DataFrame()
    # msfdata['RawFileName']    = list(set(usrdata.df.SpectrumFile.tolist()))
    # msfdata['RawFileName']    = sorted(usrdata.df.SpectrumFile.unique())
    msfdata["RawFileName"] = sorted(psm_data.SpectrumFile.fillna("").unique())
    msfdata["EXPRecNo"] = usrdata.recno
    msfdata["EXPRunNo"] = usrdata.runno
    msfdata["EXPSearchNo"] = usrdata.searchno
    msfdata["AddedBy"] = usrdata.added_by
    msfdata["CreationTS"] = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    msfdata["ModificationTS"] = datetime.now().strftime("%m/%d/%Y %H:%M:%S")

    if len(psm_data.SpectrumFile.unique()) > 300:
        warn(
            f"{psm_data.SpectrumFile.unique()} spectrum files found, too many for calculation. Skipping"
        )
        return msfdata

    summary_info = msfdata.apply(
        lambda x: _spectra_summary(
            x["RawFileName"],
            # usrdata.df),
            psm_data,
        ),
        axis=1,
    )

    (
        msfdata["RTmin_min"],
        msfdata["RTmin_max"],
        msfdata["IonScore_min"],
        msfdata["IonScore_max"],
        msfdata["qValue_min"],
        msfdata["qValue_max"],
        msfdata["PEP_min"],
        msfdata["PEP_max"],
        msfdata["Area_min"],
        msfdata["Area_max"],
        msfdata["PSMCount"],
        msfdata["DeltaMassPPM_med"],
    ) = zip(*summary_info)

    rawfile_info = msfdata.apply(
        lambda x: _get_rawfile_info(usrdata.rawfiledir, x["RawFileName"]), axis=1
    )

    msfdata["RawFileSize"], msfdata["RawFileTS"] = zip(*rawfile_info)

    return msfdata


def get_gid_ignore_list(inputfile):
    """Input a file with a list of geneids to ignore when normalizing across taxa
    Each line should have 1 geneid on it.
    Use '#' at the start of the line for comments
    Output a list of geneids to ignore.
    """
    # Don't convert GIDs to ints,
    # GIDs are not ints for the input data
    return [x.strip() for x in open(inputfile, "r") if not x.strip().startswith("#")]


def _format_peptideinfo(row):
    # if len(row) == 0:
    #     return ("", 0, "", 0, "", 0, "", 0, "", 0, ())

    def filterfunc(x):
        return any([pd.isna(x) or x == -1 or x == "-1" or x == ""])

    # def format_field(field_name):
    #     unique_values = {x for x in row[field_name] if not filterfunc(x)}
    #     joined_values = SEP.join(str(x) for x in unique_values)
    #     num_unique = len(unique_values)
    #     return joined_values, num_unique
    def format_field(field_name, row):
        unique_values = {x for x in row[field_name] if not filterfunc(x)}
        joined_values = SEP.join(str(x) for x in unique_values)
        num_unique = len(unique_values)
        return joined_values, num_unique

    mapping_table = {  # there are always exceptions
        "geneid": "GeneID",
        "taxon": "TaxonID",
        "taxonid": "TaxonID",
        "symbol": "GeneSymbol",
        "ref": "ProteinRef",
        "gi": "ProteinGI",
        "hid": "HID",
    }
    result = {}
    for header in row.columns:
        if header.lower() == "sequence":
            continue  # don't need this
        if header != "capacity":
            joined, count = format_field(header, row)
            header_name = mapping_table.get(header.lower(), header)
            result[f"{header_name}s_All"] = joined
            result[f"{header_name}Count_All"] = count
        elif header == "capacity":
            capacity_info = SEP.join(
                str(x) for x in row["capacity"] if not filterfunc(x)
            )
            result["GeneCapacities"] = capacity_info

    result_series = pd.Series(result)

    return result_series

    # result = (
    #     # ','.join(row['GeneID'].dropna().unique()),
    #     SEP.join(str(x) for x in set(row["geneid"]) if not filterfunc(x)),
    #     row["geneid"].replace("", np.nan).nunique(dropna=True),
    #     # ','.join(row['TaxonID'].dropna().unique()),
    #     SEP.join(str(x) for x in set(row["taxon"]) if not filterfunc(x)),
    #     row["taxon"].replace("", np.nan).replace("-1", np.nan).nunique(dropna=True),
    #     # ','.join(row['ProteinGI'].dropna().unique()),
    #     SEP.join(str(x) for x in set(row["gi"]) if not filterfunc(x)),
    #     row["gi"].replace("", np.nan).replace("-1", np.nan).nunique(dropna=True),
    #     SEP.join(str(x) for x in set(row["ref"])),
    #     row["ref"].replace("", np.nan).replace("-1", np.nan).nunique(dropna=True),
    #     # ','.join(row['HomologeneID'].dropna().unique()),
    #     SEP.join(str(x) for x in set(row["homologene"]) if not filterfunc(x)),
    #     row["homologene"]
    #     .replace("", np.nan)
    #     .replace("-1", np.nan)
    #     .nunique(dropna=True),
    #     SEP.join(str(x) for x in row["capacity"] if not filterfunc(x)),
    #     # tuple(row['capacity']),
    #     # row['capacity'].mean(),
    # )
    # return result


def _extract_peptideinfo(row, database):
    return _format_peptideinfo(database.loc[row])


def extract_peptideinfo(usrdata, database):
    filter_int = partial(filter, lambda x: x.isdigit())
    to_int = partial(map, int)
    ixs = (
        usrdata.df.metadatainfo.str.strip("|")
        .str.split("|")
        .apply(filter_int)
        .apply(to_int)
        .apply(list)
        # .apply(pd.Series)
        # .stack()
        # .to_frame()
    )

    # info = ixs.apply(lambda x : _format_peptideinfo(database.loc[x])).apply(pd.Series)
    info = ixs.pipe(
        apply_by_multiprocessing,
        _extract_peptideinfo,
        func_args=(database,),
        workers=WORKERS,
    ).apply(pd.Series)

    # info.columns = [
    #     "GeneIDs_All",
    #     "GeneIDCount_All",
    #     "TaxonIDs_All",
    #     "TaxonIDCount_All",
    #     "ProteinGIs_All",
    #     "ProteinGICount_All",
    #     "ProteinRefs_All",
    #     "ProteinRefCount_All",
    #     "HIDs",
    #     "HIDCount_All",
    #     "GeneCapacities",
    # ]
    for col in (
        "TaxonIDs_All",
        "ProteinGIs_All",
        "ProteinGICount_All",
        "ProteinRefs_All",
        "ProteinRefCount_All",
        "HIDs",
        "HIDCount_All",
    ):
        if col not in info.columns:
            continue
        info[col] = info[col].astype(
            "category"
        )  # because there are many less unique values than there are records, we can generally do this
    info["TaxonIDCount_All"] = info["TaxonIDCount_All"].astype(np.int16)

    usrdata.df = usrdata.df.join(info)
    # (usrdata.df['GeneIDs_All'],
    #  usrdata.df['GeneIDCount_All'],
    #  usrdata.df['TaxonIDs_All'],
    #  usrdata.df['TaxonIDCount_All'],
    #  usrdata.df['ProteinGIs_All'],
    #  usrdata.df['ProteinGICount_All'],
    #  usrdata.df['ProteinRefs_All'],
    #  usrdata.df['ProteinRefCount_All'],
    #  usrdata.df['HIDs'],
    #  usrdata.df['HIDCount_All'],
    #  usrdata.df['GeneCapacities']) = zip(*info)
    # usrdata.df['TaxonIDs_All'] = usrdata.df['TaxonIDs_All'].dropna().astype(str)
    # usrdata.df['HIDs'] = usrdata.df['HIDs'].fillna('')

    return 0


def combine_coverage(start_end):
    start_end = sorted(copy(start_end))

    # for ix, entry in enumerate(start_end[:-1]):
    ix = 0
    while ix < len(start_end):
        try:
            entry = start_end[ix]
            next_entry = start_end[ix + 1]
        except IndexError:  # done
            break
        if entry[1] >= next_entry[0] and entry[1] <= next_entry[1]:
            start_end[ix][1] = next_entry[1]
            start_end.pop(ix + 1)
        else:
            ix += 1

    return start_end


def _calc_coverage(seqs, pepts):
    pepts = sorted(
        pepts, key=lambda x: min(y + len(x) for y in [s.find(x) for s in seqs])
    )

    coverages = list()
    for s in seqs:
        start_end = list()
        coverage = 0
        for pept in pepts:
            start = 0
            mark = s.find(pept.upper(), start)
            while mark != -1:
                start_id, end_id = mark, mark + len(pept)
                start += end_id

                for start_i, end_i in start_end:
                    if start_id < end_i and end_id > end_i:
                        start_id = end_i
                        break
                    elif start_id < start_i and end_id > start_i and end_id < end_i:
                        end_id = start_i
                        break
                    elif start_id >= start_i and end_id <= end_i:
                        start_id = end_id = 0
                        break
                    else:
                        continue

                if start_id != end_id:
                    start_end.append([start_id, end_id])
                    # start_end = combine_coverage(start_end)  # only need to do this if we updated this list
                # coverage += end_id-start_id
                mark = s.find(pept.upper(), start)

        start_end = combine_coverage(start_end)
        coverage = np.sum([x[1] - x[0] for x in start_end])
        coverages.append(coverage / len(s))
        # sum(y-x)

    if coverages:
        return np.mean(coverages)
    else:
        print("Warning, GeneID", row["GeneID"], "has a coverage of 0")
        return 0


def calc_coverage_axis(row, fa, psms):
    """
    Calculates total and u2g coverage for each GeneID (row) with respect to
    reference fasta (fa) and peptide evidence (psms)
    """

    if row["GeneID"] == "-1":  # reserved for no GeneID match
        return 0, 0

    seqs = fa[fa.geneid == row["GeneID"]]["sequence"].tolist()
    if len(seqs) == 0:  # mismatch
        warn(
            "When calculating coverage, No sequence for GeneID {} not found in fasta file".format(
                row["GeneID"]
            )
        )
        return 0, 0
    pepts = row["PeptidePrint"].split("_")

    u2g_pepts = psms[
        (psms.GeneID == row["GeneID"]) & (psms.GeneIDCount_All == 1)
    ].Sequence.unique()

    return (
        _calc_coverage(seqs, pepts),
        _calc_coverage(seqs, u2g_pepts) if len(u2g_pepts) > 0 else 0,
    )


def calc_coverage(df, fa, psms):
    res = df.pipe(
        apply_by_multiprocessing,
        calc_coverage_axis,
        workers=WORKERS,
        # func_args=(fa.fillna(''), psms),
        func_args=(fa, psms),
        axis=1,
    )
    # df['Coverage'], df['Coverage_u2g'] = list(zip(res))
    df["Coverage"], df["Coverage_u2g"] = list(zip(*res))
    return df


def gene_mapper(df, other_col=None):
    if other_col is None or other_col not in df.columns:
        raise ValueError("Must specify other column")
    groupdf = df[["geneid", other_col]].drop_duplicates().groupby("geneid")

    # d = {k: SEP.join(filter(None, str(v))) for k, v in groupdf[other_col]}
    d = {k: SEP.join(filter(None, map(str, v))) for k, v in groupdf[other_col]}

    return d


def gene_taxon_mapper(df):
    """Returns a dictionary with mapping:
    gene -> taxon
    Input is the metadata extracted previously"""
    return gene_mapper(df, "taxon")


def gene_symbol_mapper(df):
    """Returns a dictionary with mapping:
    gene -> taxon
    Input is the metadata extracted previously"""
    return gene_mapper(df, "symbol")


def gene_desc_mapper(df):
    """Returns a dictionary with mapping:
    gene -> taxon
    Input is the metadata extracted previously"""
    return gene_mapper(df, "description")


def gene_hid_mapper(df):
    """Returns a dictionary with mapping:
    gene -> taxon
    Input is the metadata extracted previously"""
    return gene_mapper(df, "homologene")


def gene_protgi_mapper(df):
    """Returns a dictionary with mapping:
    gene -> taxon
    Input is the metadata extracted previously"""
    return gene_mapper(df, "gi")


def gene_protref_mapper(df):
    """Returns a dictionary with mapping:
    gene -> taxon
    Input is the metadata extracted previously"""
    return gene_mapper(df, "ref")


def assign_IDG(df, filtervalues=None):
    filtervalues = filtervalues or dict()
    ion_score_bins = filtervalues.get("ion_score_bins", (10, 20, 30))
    df["PSM_IDG"] = (
        pd.cut(
            df["IonScore"].fillna(0),
            # bins=(0, *ion_score_bins, np.inf),
            bins=(-np.inf,) + tuple(ion_score_bins) + (np.inf,),
            labels=[7, 5, 3, 1],
            include_lowest=True,
            right=False,
        )
        .astype("int")
        .fillna(7)
    )
    df.loc[df["q_value"] > 0.01, "PSM_IDG"] += 1
    df.loc[(df["IonScore"].isna() | df["q_value"].isna()), "PSM_IDG"] = 9
    return df


def make_seqlower(usrdata, col="Sequence"):
    """Make a new column called sequence_lower from a DataFrame"""
    usrdata["sequence_lower"] = usrdata[col].str.lower()
    return


def peptidome_matcher(usrdata, ref_dict):
    if not ref_dict:
        return usrdata

    ref_dict_filtered = ref_dict
    pmap = partial(map, str)
    result = (
        usrdata.Sequence.str.upper()
        .map(ref_dict)
        .fillna("")
        .map(pmap)
        .map("|".join)
        .add("|")
    )
    usrdata["metadatainfo"] += result
    return usrdata


def redundant_peaks(usrdata):
    """Remove redundant, often ambiguous peaks by keeping the peak
    with the highest ion score"""
    peaks = usrdata.sort_values(by="IonScore", ascending=False).drop_duplicates(
        subset=["SpectrumFile", "SequenceModi", "Charge", "PrecursorArea"]
    )
    peaks["Peak_UseFLAG"] = 1
    # peaks['Peak_UseFLAG'] = True
    usrdata = usrdata.join(peaks["Peak_UseFLAG"])
    usrdata["Peak_UseFLAG"] = usrdata.Peak_UseFLAG.fillna(0).astype(np.int8)

    _nredundant = len(usrdata) - len(peaks)
    # usrdata['Peak_UseFLAG'] = usrdata.Peak_UseFLAG.fillna(False)
    logging.info(f"Redundant peak areas removed : {_nredundant}")
    return usrdata


def sum_area(df):
    """Sum the area of similar peaks
    New column SequenceArea is created
    """
    df["Sequence_set"] = df["Sequence"].apply(lambda x: tuple(set(list(x))))
    summed_area = (
        df.query("Peak_UseFLAG==1")
        # .filter(items=['SequenceModi', 'Charge', 'PrecursorArea'])
        .groupby(["SequenceModi", "Charge"])
        .agg({"PrecursorArea_split": "sum"})
        .reset_index()
        .rename(columns={"PrecursorArea_split": "SequenceArea"})
    )
    df = df.merge(summed_area, how="left", on=["SequenceModi", "Charge"])
    return df


def auc_reflagger(df):
    """Remove duplicate sequence areas"""
    # usrdata['Sequence_set'] = usrdata['Sequence'].apply(lambda x: tuple(set(list(x))))
    no_dups = (
        df.sort_values(
            by=[
                "SequenceModi",
                "Charge",
                "SequenceArea",
                "PSM_IDG",
                "IonScore",
                "PEP",
                "q_value",
            ],
            ascending=[1, 1, 0, 1, 0, 1, 1],
        )
        .drop_duplicates(
            subset=[
                "SequenceArea",
                "Charge",
                "SequenceModi",
            ]
        )
        .assign(AUC_reflagger=True)
    )
    df = df.join(no_dups[["AUC_reflagger"]]).assign(
        AUC_reflagger=lambda x: (x["AUC_reflagger"].fillna(0).astype(np.int8))
    )
    return df


def export_metadata(
    program_title="version",
    usrdata=None,
    matched_psms=0,
    unmatched_psms=0,
    usrfile="file",
    taxon_totals=dict(),
    outname=None,
    outpath=".",
    **kwargs,
):
    """Update iSPEC database with some metadata information"""
    print("{} | Exporting metadata".format(time.ctime()))
    # print('Number of matched psms : ', matched_psms)
    d = dict(
        version=program_title,
        searchdb=usrdata.searchdb,
        filterstamp=usrdata.filterstamp,
        matched_psms=matched_psms,
        unmatched_psms=unmatched_psms,
        inputname=usrdata.datafile,
        hu=taxon_totals.get("9606", 0),
        mou=taxon_totals.get("10090", 0),
        gg=taxon_totals.get("9031", 0),
        recno=usrdata.recno,
        runno=usrdata.runno,
        searchno=usrdata.searchno,
    )
    with open(os.path.join(outpath, outname), "w") as f:
        json.dump(d, f)


def split_on_geneid(df):
    """Duplicate psms based on geneids. Areas of each psm is recalculated based on
    unique peptides unique for its particular geneid later.
    """
    oriflag = lambda x: 1 if x[-1] == 0 else 0
    glstsplitter = (
        df["GeneIDs_All"]
        .str.split(SEP)
        .apply(pd.Series, 1)
        .stack()
        .to_frame(name="GeneID")
        .assign(oriFLAG=lambda x: x.index.map(oriflag))
    )

    glstsplitter.index = glstsplitter.index.droplevel(-1)  # get rid of
    # multi-index
    df = df.join(glstsplitter).reset_index()
    df["GeneID"] = df.GeneID.fillna("-1")
    df.loc[df.GeneID == "", "GeneID"] = "-1"
    df["GeneID"] = df.GeneID.fillna("-1")
    # df['GeneID'] = df.GeneID.astype(int)
    df["GeneID"] = df.GeneID.astype(str)
    return df


def rank_peptides(df, area_col, ranks_only=False):
    """Rank peptides here
    area_col is sequence area_calculator
    ranks_only returns just the ranks column. This does not reset the original index
    """
    df = df.sort_values(
        by=[
            "GeneID",
            area_col,
            "SequenceModi",
            "Charge",
            "PSM_IDG",
            "IonScore",
            "PEP",
            "q_value",
        ],
        ascending=[1, 0, 0, 1, 1, 0, 1, 1],
    )
    if not ranks_only:  # don't reset index for just the ranks
        df.reset_index(inplace=True)  # drop=True ?
    df['Modifications'] = df.Modifications.fillna("")
    # df[area_col].fillna(0, inplace=True)  # must do this to compare
    df[area_col] = df[area_col].fillna(0)  # must do this to compare
    # nans
    ranks = (
        df[(df.AUC_UseFLAG == 1) & (df.PSM_UseFLAG == 1) & (df.Peak_UseFLAG == 1)]
        .groupby(["GeneID", "LabelFLAG"])
        .cumcount()
        + 1
    )  # add 1 to start the peptide rank at 1, not 0
    ranks.name = "PeptRank"
    if ranks_only:
        return ranks
    df = df.join(ranks)
    return df


def flag_AUC_PSM(
    df, fv, contaminant_label="__CONTAMINANT__", phospho=False, acetyl=False
):
    if fv["pep"] == "all":
        fv["pep"] = float("inf")
    if fv["idg"] == "all":
        fv["idg"] = float("inf")
    df["AUC_UseFLAG"] = 1
    df["PSM_UseFLAG"] = 1
    df.loc[
        (df["Charge"] < fv["zmin"]) | (df["Charge"] > fv["zmax"]),
        ["AUC_UseFLAG", "PSM_UseFLAG"],
    ] = 0

    df.loc[df["SequenceModiCount"] > fv["modi"], ["AUC_UseFLAG", "PSM_UseFLAG"]] = 0

    df.loc[
        (df["IonScore"].isnull() | df["q_value"].isnull()),
        ["AUC_UseFLAG", "PSM_UseFLAG"],
    ] = (1, 0)

    df.loc[df["IonScore"] < fv["ion_score"], ["AUC_UseFLAG", "PSM_UseFLAG"]] = 0

    df.loc[df["q_value"] > fv["qvalue"], ["AUC_UseFLAG", "PSM_UseFLAG"]] = 0

    df.loc[df["PEP"] > fv["pep"], ["AUC_UseFLAG", "PSM_UseFLAG"]] = 0
    df.loc[df["PSM_IDG"] > fv["idg"], ["AUC_UseFLAG", "PSM_UseFLAG"]] = 0

    if df["PSMAmbiguity"].dtype == str:
        df.loc[
            (df["Peak_UseFLAG"] == 0)
            & (df["PSMAmbiguity"].str.lower() == "unambiguous"),
            ["AUC_UseFLAG", "PSM_UseFLAG"],
        ] = 1
        df.loc[
            (df["Peak_UseFLAG"] == 0)
            & (df["PSMAmbiguity"].str.lower() != "unambiguous"),
            ["AUC_UseFLAG", "PSM_UseFLAG"],
        ] = 0
    elif any(df["PSMAmbiguity"].dtype == x for x in (int, float)):
        df.loc[
            (df["Peak_UseFLAG"] == 0) & (df["PSMAmbiguity"] == 0),
            ["AUC_UseFLAG", "PSM_UseFLAG"],
        ] = 1
        df.loc[
            (df["Peak_UseFLAG"] == 0) & (df["PSMAmbiguity"] != 0),
            ["AUC_UseFLAG", "PSM_UseFLAG"],
        ] = 0

    df.loc[df["AUC_reflagger"] == 0, "AUC_UseFLAG"] = 0
    df.loc[ df.is_decoy == True, 'AUC_UseFLAG' ] = 0
    df.loc[ df.is_decoy == True, 'PSM_USeFLAG' ] = 0

    # we don't actually want to do this here because we want the contaminant peptides (and gene products) to "soak up" value, as contaminants are meant to do
    # df.loc[
    #     df["GeneIDs_All"].fillna("").str.contains(contaminant_label),
    #     ["AUC_UseFLAG", "PSM_UseFLAG"],
    # ] = (0, 0)

    # phospho modifications are designated in SequenceModi via: XXS(pho)XX
    # similarly acetyl modifications are designated via: XXK(ace)
    if phospho:
        df.loc[
            ~df["SequenceModi"].str.contains("pho|79.966", case=True),
            ["AUC_UseFLAG", "PSM_UseFLAG"],
        ] = (0, 0)
    elif acetyl:
        df.loc[
            ~df["SequenceModi"].str.contains("ace", case=True),
            ["AUC_UseFLAG", "PSM_UseFLAG"],
        ] = (0, 0)

    return df


def gene_taxon_map(usrdata, gene_taxon_dict):
    """make 'gene_taxon_map' column per row which displays taxon for given gene"""
    usrdata["TaxonID"] = usrdata["GeneID"].map(gene_taxon_dict)
    return


def get_all_taxons(taxonidlist):
    """Return a set of all taxonids from
    usrdata.TaxonIDList"""
    taxon_ids = set(SEP.join(x for x in taxonidlist if x.strip()).split(SEP))
    return taxon_ids


def multi_taxon_splitter(taxon_ids, usrdata, gid_ignore_list, area_col):
    """Plugin for multiple taxons
    Returns a dictionary with the totals for each detected taxon"""
    taxon_totals = dict()
    for taxon in taxon_ids:
        # all_others = [x for x in taxon_ids if x != taxon]
        uniq_taxon = usrdata[
            # (usrdata._data_tTaxonIDList.str.contains(taxon)) &
            # (~usrdata._data_tTaxonIDList.str.contains('|'.join(all_others)))&
            (usrdata["AUC_UseFLAG"] == 1)
            & (usrdata["TaxonID"] == str(taxon))
            & (usrdata["TaxonIDCount_All"] == 1)
            & (~usrdata["GeneID"].isin(gid_ignore_list))
        ]
        taxon_totals[taxon] = (
            uniq_taxon[area_col] / uniq_taxon["GeneIDCount_All"]
        ).sum()

    tot_unique = sum(taxon_totals.values())  # sum of all unique
    # now compute ratio:

    for taxon in taxon_ids:
        taxon = str(taxon)
        try:
            percentage = taxon_totals[taxon] / tot_unique
        except ZeroDivisionError:
            warn(
                """This file has multiple taxa but no unique to taxa peptides.
            Please check this experiment
            """
            )
            percentage = 1
        taxon_totals[taxon] = percentage
        print(taxon, " ratio : ", taxon_totals[taxon])
        # logfile.write('{} ratio : {}\n'.format(taxon, taxon_totals[taxon]))
    return taxon_totals


# def create_df(inputdf, label, inputcol='GeneID'):
#     """Create and return a DataFrame with gene/protein information from the input
#     peptide DataFrame"""
#     return pd.DataFrame({'GeneID':
#                          list(set(inputdf[inputcol])),
#                          'EXPLabelFLAG': labelflag.get(label, label)})


# def create_df(inputdf, inputcol='GeneID'):
#     """Create and return a DataFrame with gene/protein information from the input
#     peptide DataFrame"""
#     return pd.DataFrame({'GeneID':
#                          list(set(inputdf[inputcol])),
#                          'EXPLabelFLAG': labelflag.get(label, label)})
def create_df(inputdf, inputcol="GeneID"):
    """Create and return a DataFrame with gene/protein information from the input
    peptide DataFrame"""
    return pd.DataFrame(
        {
            "GeneID": list(set(inputdf[inputcol])),
        }
    )


# def select_good_peptides(usrdata, labelix):
#     """Selects peptides of a given label with the correct flag and at least one genecount
#     The LabelFLAG is set here for TMT/iTRAQ/SILAC data.
#     """
#     temp_df = usrdata[((usrdata['PSM_UseFLAG'] == 1) | usrdata['AUC_UseFLAG'] ==1) &
#                       (usrdata['GeneIDCount_All'] > 0)].copy()  # keep match between runs
#     temp_df['LabelFLAG'] = labelix
#     return temp_df
def select_good_peptides(usrdata):
    """Selects peptides of a given label with the correct flag and at least one genecount
    The LabelFLAG is no longer needed
    """
    temp_df = usrdata[
        ((usrdata["PSM_UseFLAG"] == 1) | usrdata["AUC_UseFLAG"] == 1)
        & (usrdata["GeneIDCount_All"] > 0)
    ].copy()  # keep match between runs
    return temp_df


def get_gene_capacity(genes_df, database, col="GeneID"):
    """Get gene capcaity from the stored metadata"""
    capacity = database.groupby("geneid").capacity.mean().to_frame(name="GeneCapacity")
    genes_df = genes_df.merge(capacity, how="left", left_on="GeneID", right_index=True)
    return genes_df


def get_gene_info(genes_df, database, col="GeneID"):
    subset = ["geneid", "homologene", "description", "symbol", "taxon"]
    genecapacity = database.groupby("geneid")["capacity"].mean().rename("capacity_mean")
    geneinfo = (
        database[subset]
        .drop_duplicates("geneid")
        .set_index("geneid")
        .join(genecapacity)
        .rename(
            columns=dict(
                gi="ProteinGI",
                homologene="HomologeneID",
                taxon="TaxonID",
                description="Description",
                ref="ProteinAccession",
                symbol="GeneSymbol",
                capacity_mean="GeneCapacity",
            )
        )
    )
    # geneinfo.index = geneinfo.index.astype(str)
    # geneinfo['TaxonID'] = geneinfo.TaxonID.astype(str)
    out = genes_df.merge(geneinfo, how="left", left_on="GeneID", right_index=True)
    return out


def get_peptides_for_gene(genes_df, temp_df):
    # this is often an area where errors manifest
    if len(temp_df) == 0:
        raise ValueError("input psms data frame is empty")

    full = (
        temp_df.groupby("GeneID")["sequence_lower"].agg(
            (lambda x: frozenset(x), "nunique")
        )
        # this changed from <lambda> to <lambda_0>
        .rename(
            columns={
                "<lambda>": "PeptideSet",
                "<lambda_0>": "PeptideSet",
                "nunique": "PeptideCount",
            }
        )
        # .agg(full_op)
        .assign(PeptidePrint=lambda x: x["PeptideSet"].apply(sorted).str.join("_"))
    )
    full["PeptideSet"] = full.apply(lambda x: frozenset(x["PeptideSet"]), axis=1)

    q_uniq = "GeneIDCount_All == 1"
    q_strict = "PSM_IDG < 4"
    q_strict_u = "{} & {}".format(q_uniq, q_strict)

    try:
        uniq = (
            temp_df.query(q_uniq)
            .groupby("GeneID")["sequence_lower"]
            .agg("nunique")
            .to_frame("PeptideCount_u2g")
        )
    except IndexError:
        uniq = pd.DataFrame(columns=["PeptideCount_u2g"])

    try:
        strict = (
            temp_df.query(q_strict)
            .groupby("GeneID")["sequence_lower"]
            .agg("nunique")
            .to_frame("PeptideCount_S")
        )
    except IndexError:
        strict = pd.DataFrame(columns=["PeptideCount_S"])

    try:
        s_u2g = (
            temp_df.query(q_strict_u)
            .groupby("GeneID")["sequence_lower"]
            .agg("nunique")
            .to_frame("PeptideCount_S_u2g")
        )
    except IndexError:
        s_u2g = pd.DataFrame(columns=["PeptideCount_S_u2g"])

    result = pd.concat((full, uniq, strict, s_u2g), copy=False, axis=1).fillna(0)
    ints = [
        "" + x
        for x in (
            "PeptideCount",
            "PeptideCount_u2g",
            "PeptideCount_S",
            "PeptideCount_S_u2g",
        )
    ]
    result[ints] = result[ints].astype(int)

    genes_df = genes_df.merge(result, how="left", left_on="GeneID", right_index=True)
    return genes_df


def get_psms_for_gene(genes_df, temp_df):
    psmflag = "PSM_UseFLAG"

    total = temp_df.groupby("GeneID")[psmflag].sum()
    total.name = "PSMs"

    q_uniq = "GeneIDCount_All == 1"
    total_u2g = temp_df.query(q_uniq).groupby("GeneID")[psmflag].sum()
    total_u2g.name = "PSMs_u2g"

    q_strict = "PSM_IDG < 4"
    total_s = temp_df.query(q_strict).groupby("GeneID")[psmflag].sum()
    total_s.name = "PSMs_S"

    q_strict_u = "{} & {}".format(q_uniq, q_strict)
    total_s_u2g = temp_df.query(q_strict_u).groupby("GeneID")[psmflag].sum()
    total_s_u2g.name = "PSMs_S_u2g"

    result = (
        pd.concat((total, total_u2g, total_s, total_s_u2g), copy=False, axis=1)
        .fillna(0)
        .astype(int)
    )
    genes_df = genes_df.merge(result, how="left", left_on="GeneID", right_index=True)
    return genes_df


def calculate_full_areas(genes_df, temp_df, area_col, normalize):
    """Calculates full (non distributed ) areas for gene ids.
    calculates full, gene count normalized, unique to gene,
    and unique to gene with no miscut areas.
    """
    logging.info("Calculating peak areas")

    qstring = "AUC_UseFLAG == 1"
    full = temp_df.query(qstring).groupby("GeneID")[area_col].sum() / normalize
    full.name = "AreaSum_max"

    # full_adj = (temp_df.query(qstring)
    #             .assign(gpAdj = lambda x: x[area_col] / x['GeneIDCount_All'])
    #             .groupby('GeneID')['gpAdj']  # temp column
    #             .sum()/normalize
    #             )
    # full_adj.name = 'AreaSum_gpcAdj'

    # qstring_s = qstring + ' & IDG < 4'
    # strict = temp_df.query(qstring_s).groupby('GeneID')[area_col].sum()
    # strict.name = ''

    qstring_u = qstring + " & GeneIDCount_All == 1"
    uniq = temp_df.query(qstring_u).groupby("GeneID")[area_col].sum() / normalize
    uniq.name = "AreaSum_u2g_all"

    qstring_u0 = qstring_u + " & MissedCleavages == 0"
    uniq_0 = temp_df.query(qstring_u0).groupby("GeneID")[area_col].sum() / normalize
    uniq_0.name = "AreaSum_u2g_0"
    result = pd.concat((full, uniq, uniq_0), copy=False, axis=1).fillna(0)
    genes_df = genes_df.merge(result, how="left", left_on="GeneID", right_index=True)
    return genes_df


def _distribute_area(
    inputdata, genes_df, area_col, taxon_totals=None, taxon_redistribute=True
):
    """Row based normalization of PSM area (mapped to a specific gene).
    Normalization is based on the ratio of the area of unique peptides for the
    specific gene to the sum of the areas of the unique peptides for all other genes
    that this particular peptide also maps to.
    """
    # if inputdata.AUC_UseFLAG == 0:
    #     return 0
    inputvalue = inputdata[area_col]
    geneid = inputdata["GeneID"]
    gene_inputdata = genes_df.query("GeneID == @geneid")
    u2g_values = gene_inputdata["AreaSum_u2g_all"].values

    if len(u2g_values) == 1:
        u2g_area = u2g_values[0]  # grab u2g info, should always be
    # of length 1
    elif len(u2g_values) > 1:
        warn(
            "DistArea is not singular at GeneID : {}".format(
                datetime.now(), inputdata["GeneID"]
            )
        )
        distArea = 0
        # this should never happen (and never has)
    else:
        distArea = 0
        print("No distArea for GeneID : {}".format(inputdata["GeneID"]))
    # taxon_ratio = taxon_totals.get(inputdata.gene_taxon_map, 1)

    if u2g_area != 0:
        totArea = 0
        gene_list = inputdata.GeneIDs_All.split(SEP)
        all_u2gareas = (
            genes_df[genes_df["GeneID"].isin(gene_list)]
            .query("PeptideCount_u2g > 0")  # all geneids with at least 1 unique pept
            .AreaSum_u2g_all
        )
        if len(all_u2gareas) > 1 and any(x == 0 for x in all_u2gareas):
            # special case with multiple u2g peptides but not all have areas, rare but does happen
            u2g_area = 0  # force to distribute by gene count (and taxon percentage if appropriate)
        else:
            totArea = all_u2gareas.sum()
            distArea = (u2g_area / totArea) * inputvalue
        # ratio of u2g peptides over total area
    elif all(gene_inputdata.IDSet == 3):
        return 0
    if u2g_area == 0:  # no uniques, normalize by genecount
        taxon_percentage = taxon_totals.get(str(inputdata.TaxonID), 1)
        distArea = inputvalue
        if taxon_percentage < 1:
            distArea *= taxon_percentage
        gpg_selection = genes_df.GPGroup == gene_inputdata.GPGroup.values[0]
        try:
            if taxon_redistribute:
                taxonid_selection = genes_df.TaxonID == gene_inputdata.TaxonID.values[0]
                distArea /= len(genes_df[(gpg_selection) & (taxonid_selection)])
            else:
                distArea /= len(genes_df[(gpg_selection)])
        except ZeroDivisionError:
            pass

    return distArea


def distribute_area(temp_df, genes_df, area_col, taxon_totals, taxon_redistribute=True):
    """Distribute psm area based on unique gene product area
    Checks for AUC_UseFLAG==1 for whether or not to use each peak for quantification
    """

    q = "AUC_UseFLAG == 1 & GeneIDCount_All > 1"
    distarea = "PrecursorArea_dstrAdj"
    temp_df[distarea] = 0
    # temp_df[distarea] = (temp_df.query(q)
    #                      .apply(
    #                          _distribute_area, args=(genes_df,
    #                                                      area_col,
    #                                                      taxon_totals,
    #                                                      taxon_redistribute),
    #                          axis=1)
    # )
    temp_df[distarea] = temp_df.query(q).pipe(
        apply_by_multiprocessing,
        _distribute_area,
        workers=WORKERS,
        func_args=(genes_df, area_col, taxon_totals, taxon_redistribute),
        axis=1,
    )

    one_id = (temp_df.GeneIDCount_All == 1) & (temp_df.AUC_UseFLAG == 1)
    temp_df.loc[one_id, distarea] = temp_df.loc[one_id, area_col]
    # temp_df[distarea].fillna(0, inplace=True)

    return


def _set2_or_3(row, genes_df, allsets):
    # print(row.GeneID)
    # if row.GeneID == 'Lactobacillus1142' or row.GeneID == 'Alistipes42':

    peptset = row.PeptideSet
    # allsets = genes_df.PeptideSet.unique()  # calculate outside this function for performance boost
    if six.PY2 and any(set(peptset) < x for x in allsets):
        return 3

    elif six.PY3 and any(peptset < allsets):
        return 3

    # check if is set 3 across multiple genes, or is set2
    gid = row.GeneID

    # sel = genes_df[ (genes_df.IDSet == 1) &
    #                 (genes_df.PeptideSet & peptset) ].query('GeneID != @gid')

    # sel = genes_df[(genes_df.PeptideSet & peptset) ].query('GeneID != @gid') # doesn't work anymore??
    # this fix does the equivalent? Believe so
    sel = genes_df[
        (genes_df.PeptideSet.apply(lambda x: x.intersection(peptset))).apply(bool)
    ].query("GeneID != @gid")
    sel_idset1 = sel.query("IDSet == 1")

    in_pop = sel.PeptideSet
    in_pop_set1 = sel_idset1.PeptideSet

    in_row = sel.apply(lambda x: peptset - x["PeptideSet"], axis=1)

    if in_pop.empty:
        in_pop_all = set()
    else:
        in_pop_all = set(in_pop.apply(tuple).apply(pd.Series).stack().unique())
    # except Exception as e:

    if not in_pop_set1.empty:
        in_pop_all_set1 = set(
            in_pop_set1.apply(tuple).apply(pd.Series).stack().unique()
        )
    else:
        in_pop_all_set1 = set()

    diff = peptset - in_pop_all  # check if is not a subset of anything

    diff_idset1 = peptset - in_pop_all_set1  # check if is not a subset of set1 ids

    if len(diff_idset1) == 0:  # is a subset of idset1 ids
        return 3

    elif len(diff) > 0:  # is not a subset of anything
        return 2

    else:
        sel_not_idset1 = sel.query("IDSet != 1")

        if any(sel_not_idset1.PeptideSet == peptset):
            return (
                2  # shares all peptides with another, and is not a subset of anything
            )

        # need to check if is a subset of everything combined, but not a subset of one thing
        # ex:
        #        PEPTIDES
        # row    =  A   B
        # match1 =  A
        # match2 =      B
        if all((peptset - sel.PeptideSet).apply(bool)) and not all(
            (sel_not_idset1.PeptideSet - peptset).apply(bool)
        ):
            return 2
        else:
            pept_lengths = sel_not_idset1.PeptideSet.apply(len)
            if len(peptset) >= pept_lengths.max():
                return 2
            else:
                return 3

            # len_shared = sel_not_idset1.PeptideSet.apply(lambda x: x & peptset).apply(len)
            # max_shared = len_shared.max()
            # all_shared_pepts = (set([x for y in sel_not_idset1.PeptideSet.values for x in y])
            #                     & peptset)

    return 3


class _DummyDataFrame:
    def eat_args(self, *args, **kwargs):
        return None

    def __getattr__(self, name):
        if name not in self.__dict__:
            return self.eat_args


def check_length_in_pipe(df):
    """Checks the length of a DataFrame in a pipe
    and if zero returns an object to suppress all errors,
    just returning None (ideally)
    """
    if len(df) == 0:
        return _DummyDataFrame()
    return df


def assign_gene_sets(genes_df, temp_df):
    logging.info("Assigning gene sets")
    all_ = genes_df.PeptideSet.unique()
    allsets = genes_df.PeptideSet.unique()
    genes_df.loc[genes_df.PeptideCount_u2g > 0, "IDSet"] = 1
    genes_df.loc[genes_df.PeptideCount_u2g == 0, "IDSet"] = (
        genes_df.query("PeptideCount_u2g == 0").pipe(check_length_in_pipe)
        # .apply(_set2_or_3, args=(genes_df, allsets), axis=1))
        .pipe(
            apply_by_multiprocessing,
            _set2_or_3,
            genes_df=genes_df,
            allsets=allsets,
            axis=1,
            workers=WORKERS,
        )
        # axis=1, workers=1)
    )
    genes_df["IDSet"] = genes_df["IDSet"].fillna(3).astype(np.int8)
    # if u2g count greater than 0 then set 1
    gpg = temp_df.query("PSM_IDG>0").groupby("GeneID").PSM_IDG.min().rename("IDGroup")
    gpg_u2g = (
        temp_df.query("GeneIDCount_All==1")
        .query("PSM_IDG>0")
        .groupby("GeneID")
        .PSM_IDG.min()
        .rename("IDGroup_u2g")
    )
    gpgs = (
        pd.concat([gpg, gpg_u2g], axis=1)
        .fillna(0)
        .astype(np.int8)
        # .assign(GeneID = lambda x: x.index)
        # .reset_index(drop=True)
    )
    genes_df = pd.merge(genes_df, gpgs, left_on="GeneID", how="left", right_index=True)
    return genes_df


def calculate_gene_dstrarea(genes_df, temp_df, normalize=1):
    """Calculate distributed area for each gene product"""
    result = (
        temp_df.query("AUC_UseFLAG == 1")
        .groupby("GeneID")["PrecursorArea_dstrAdj"]
        .sum()
        .divide(normalize)
        .to_frame(name="AreaSum_dstrAdj")
    )
    genes_df = genes_df.merge(result, how="left", left_on="GeneID", right_index=True)
    genes_df.loc[genes_df["IDSet"] == 3, "AreaSum_dstrAdj"] = 0
    return genes_df


def calculate_gene_razorarea(genes_df, temp_df, normalize):
    """Calculate razor area for each gene product"""

    separate_groups = lambda gpg_all: set(
        float(x.strip()) for z in (y.split(SEP) for y in gpg_all.values) for x in z
    )

    def razor_area(temp_df, genes_df):
        gid = temp_df.GeneID
        gpgs = genes_df[genes_df.GeneID == gid].GPGroups_All.pipe(separate_groups)
        if len(gpgs) == 0:
            return 0
        allgenes = genes_df[genes_df.GPGroup.isin(gpgs)]
        max_uniq = allgenes.PeptideCount_u2g.max()
        gene_selection = allgenes[allgenes.PeptideCount_u2g == max_uniq]
        if len(gene_selection) != 1 or max_uniq == 0:  # no uniques
            return 0
        if gid == gene_selection.GeneID.values[0]:
            return temp_df.SequenceArea
        else:
            return 0

    temp_df["RazorArea"] = 0
    q = "AUC_UseFLAG == 1 & GeneIDCount_All > 1"
    temp_df["RazorArea"] = temp_df.query(q).apply(razor_area, args=(genes_df,), axis=1)
    one_id = temp_df.GeneIDCount_All == 1
    temp_df.loc[one_id, "RazorArea"] = temp_df.loc[one_id, "SequenceArea"]
    result = temp_df.groupby("GeneID")["RazorArea"].sum() / normalize
    result = (
        temp_df.groupby("GeneID")["RazorArea"]
        .sum()
        .divide(normalize)
        .to_frame(name="AreaSum_razor")
    )
    genes_df = genes_df.merge(result, how="left", left_on="GeneID", right_index=True)
    return genes_df


def set_gene_gpgroups(genes_df, temp_df):
    """Assign GPGroups"""

    logging.info("Assigning gpgroups")

    genes_df.sort_values(by=["PSMs"], ascending=False, inplace=True)
    genes_df.reset_index(inplace=True)

    set1s = genes_df.query("IDSet==1")

    gpg_vals = range(1, len(set1s) + 1)
    set1_gpgs = pd.Series(data=gpg_vals, index=set1s.index, name="GPGroup").to_frame()

    next_gpg = len(set1_gpgs) + 1
    set2_filtered = genes_df.query("IDSet==2")
    if len(set2_filtered) > 0:
        # if len(set2_filts = (set2_filtered
        set2s = set2_filtered.groupby("PeptideSet").first().index
        set2_gpgdict = pd.Series(
            index=set2s, data=range(next_gpg, next_gpg + len(set2s))
        ).to_dict()
        genes_df["GPGroup"] = genes_df.PeptideSet.map(set2_gpgdict)
    else:
        genes_df["GPGroup"] = [np.nan] * len(genes_df)
    genes_df.loc[set1_gpgs.index, "GPGroup"] = set1_gpgs

    # (temp_df[['Sequence', 'GeneIDs_All']]
    #  .drop_duplicates('Sequence')
    #  .assign(geneids = lambda x: (x['GeneIDs_All'].str.split(',')))
    # )

    gene_pept_mapping = defaultdict(set)
    pept_group_mapping = defaultdict(set)
    valid_gpgroups = genes_df[~(genes_df.GPGroup.isnull())]
    for ix, row in valid_gpgroups.iterrows():
        for pept in row["PeptideSet"]:
            gene_pept_mapping[row.GeneID].add(pept)
            pept_group_mapping[pept].add(row.GPGroup)
    for ix, row in genes_df[genes_df.GPGroup.isnull()].iterrows():  # need to add set3s
        for pept in row["PeptideSet"]:
            gene_pept_mapping[row.GeneID].add(pept)

    def gpg_all(gid, gene_pept_mapping, pept_group_mapping):
        gpgs = set()
        for pept in gene_pept_mapping.get(gid):
            mapping = pept_group_mapping.get(pept)
            if mapping is None:
                continue
            gpgs |= pept_group_mapping.get(pept)
        return SEP.join(str(int(x)) for x in sorted(gpgs))

    genes_df["GPGroups_All"] = genes_df.apply(
        lambda x: gpg_all(x["GeneID"], gene_pept_mapping, pept_group_mapping), axis=1
    )
    # do not need to multiprocess this
    # genes_df['GPGroups_All'] = genes_df.GeneID.pipe(apply_by_multiprocessing,
    #                                                 gpg_all,
    #                                                 func_args=(gene_pept_mapping, pept_group_mapping),
    #                                                 workers=WORKERS,
    # )

    genes_df["GPGroup"] = genes_df["GPGroup"].fillna(0).astype(int)
    # genes_df['GPGroup'].replace(to_replace='', value=float('NaN'),
    #                                 inplace=True)  # can't sort int and
    # strings, convert all strings to NaN
    return genes_df


def get_labels(usrdata, labels, labeltype="none"):
    # TODO simplify
    '""labels is a dictionary of lists for each label type""'
    if labeltype == "none":  # label free
        return ["none"]
    elif labeltype == "SILAC":
        return usrdata.LabelFLAG.unique()
    mylabels = labels.get(labeltype)
    if mylabels is None:
        return ["none"]
    included_labels = [label for label in mylabels if label in usrdata.columns]
    return included_labels


def regularize_filename(f):
    """
    regularize filename so that it's valid on windows
    """
    invalids = r'< > : " / \ | ? *'.split(" ")
    # invalids = r'<>:"/\\\|\?\*'
    out = str(f)
    for i in invalids:
        out = out.replace(i, "_")
    return out


def split_multiplexed(usrdata, labeltypes, exp_labeltype="none", tmt_reference=None ):
    df = usrdata.df

    if exp_labeltype == "none":
        df["LabelFLAG"] = 0
        df["PrecursorArea_split"] = df["PrecursorArea"]
        out = os.path.join(
            usrdata.outdir, usrdata.output_name(str(0) + "_psms", ext="tsv")
        )
        df["ReporterIntensity"] = np.NAN
        quant_cols = [x for x in DATA_QUANT_COLS if x in df]
        # df[quant_cols].to_csv(out, index=False, encoding="utf-8", sep="\t")
        return {0: df[quant_cols]}

    # SILAC
    if exp_labeltype == "SILAC":
        output = dict()
        for iso_label in labeltypes:  # convert labeltype to lower for finding
            if iso_label == 0:
                subdf = df[~df["SequenceModi"].str.contains("Label")].copy()
            else:
                subdf = df[
                    df["SequenceModi"].str.contains(iso_label.lower(), regex=False)
                ].copy()

            subdf["LabelFLAG"] = 0
            subdf["ReporterIntensity"] = np.NAN
            subdf["PrecursorArea_split"] = subdf[
                "PrecursorArea"
            ]  # no splitting on reporter ion

            # outname = str(iso_label).replace(':', '_')
            outname = regularize_filename(str(iso_label))
            out = os.path.join(
                usrdata.outdir, usrdata.output_name(outname + "_psms", ext="tsv")
            )
            quant_cols = [x for x in DATA_QUANT_COLS if x in subdf]
            # subdf[quant_cols].to_csv(out, index=False, encoding="utf-8", sep="\t")
            output[iso_label] = subdf[quant_cols]
        return output

    # iTRAQ/TMT
    # output = list()
    output = dict()

    # TMTPro
    query = "{}|229|304.207|608.414".format(
        exp_labeltype.lower()
    )  # exp_labeltype is a string like "TMT" or "none"
    df["LabelFLAG"] = np.nan
    # assign labeltype
    _how_many_seqmodis_are_modified = df.loc[
        df["SequenceModi"].str.contains(query)
    ].pipe(len)
    if _how_many_seqmodis_are_modified == 0 and exp_labeltype == "TMT":
        warn(
            "labeltype is set to `TMT` yet TMT modifications detected, will assume all are modified"
        )
        # df.loc[~df["SequenceModi"].str.contains(query), "LabelFLAG"] = 0
    else:
        df.loc[~df["SequenceModi"].str.contains(query), "LabelFLAG"] = 0

    #
    # scale labeltypes

    with_reporter = df[
        (df["SequenceModi"].str.contains(query, case=True))
        | (
            df.SequenceModi.str.startswith("C")
            & (df.SequenceModi.str.contains("361.228|286.18"))
        )
        | (
            df.SequenceModi.str.startswith("K")
            & (df.SequenceModi.str.contains("608.414|458"))
        )
    ].copy()

    if usrdata.labeltype == "TMT" and len(with_reporter) == 0:
        warn(
            "No TMT modifications detected, will assume all are statically modified"
        )
        with_reporter = df.copy()

    with_reporter['total_reporter_intensity'] = with_reporter[labeltypes].sum(1)


    for label in labeltypes:
        # we add this to take care fo cystines at the n terminus with carbamidomethylation and TMT10/Pro
        # import ipdb; ipdb.set_trace()
        reporter_area = (
            with_reporter["PrecursorArea"]
            * with_reporter[label]
            / with_reporter["total_reporter_intensity"]
        )

        # now do something with tmt_reference if given and specified. we scale by the ref intensity.
        # new_area_col = '' + area_col + '_split'
        reporter_area.name = "PrecursorArea_split"
        temp_df = df.join(
            reporter_area, how="right"
        )  # only keep peptides with good reporter ion

        temp_df["LabelFLAG"] = label
        # temp_df['ReporterIntensity'] = with_reporter[label]
        temp_df = temp_df.join(with_reporter[label].to_frame("ReporterIntensity"))
        # temp_df['PrecursorArea_Split'].fillna(temp_df['PrecursorArea'], inplace=True)

        quant_cols = [x for x in DATA_QUANT_COLS if x in temp_df]
        out = os.path.join(
            usrdata.outdir, usrdata.output_name(str(label) + "_psms", ext="tsv")
        )
        output[label] = temp_df[quant_cols]
        # output[label] = out

    # output.append( df.query('LabelFLAG==0') )

    ## really not needed:
    # out = os.path.join(usrdata.outdir, usrdata.output_name(str(0)+'_psms', ext='tab'))
    # df.loc[ df.LabelFLAG==0, 'PrecursorArea_split' ] = 0
    # nolabel = df.query('LabelFLAG==0')
    # nolabel.to_csv(out, index=False, encoding='utf-8', sep='\t')
    # output[0] = out

    return output


def redistribute_area_isobar(temp_df, label, labeltypes, area_col, labeltype):
    """for tmt/itraq"""
    # with_reporter = temp_df[temp_df['QuanUsage'] == 'Use']
    q = ".*{}.*".format(labeltype.lower())
    with_reporter = temp_df[temp_df["SequenceModi"].str.contains(q)]
    reporter_area = (
        with_reporter[label]
        * with_reporter[area_col]
        / with_reporter[labeltypes].sum(1)
    )
    new_area_col = "" + area_col + "_split"
    reporter_area.name = new_area_col
    temp_df = temp_df.join(
        reporter_area, how="right"
    )  # only keep peptides with good reporter ion
    temp_df[new_area_col].fillna(temp_df[area_col], inplace=True)
    return temp_df, new_area_col


def concat_isobar_output(
    rec, run, search, outdir, outf, cols=None, labeltype=None, datatype="e2g"
):
    if labeltype == "SILAC":  # do not have digits assigned yet for different labels
        pat = re.compile(
            "^{}_{}_{}_{}_(Label)?.*_{}.tsv".format(
                rec, run, search, labeltype, datatype
            )
        )
    else:
        pat = re.compile(
            "^{}_{}_{}_{}_.*_\d+_{}.tsv".format(rec, run, search, labeltype, datatype)
        )
    files = list()
    for entry in os.scandir(outdir):
        if entry.is_file() and pat.search(entry.name):
            files.append(os.path.join(outdir, entry.name))
    files = sorted(files)
    if len(files) == 0:
        logging.warning("No output for {}_{}_{}".format(rec, run, search))
        return

    df = pd.concat((pd.read_table(f) for f in files))
    if all(x in df for x in ("LabelFLAG", "GeneID")):
        df = df.sort_values(by=["LabelFLAG", "GeneID"])
    # outf = '{}_{}_{}_{}_all_{}.tab'.format(rec, run, search, labeltype, datatype)
    # outf = '{}_{}_{}_{}_{}.tab'.format(rec, run, search, labeltype, datatype)

    if datatype == "e2g" and cols is None:
        # cols = E2G_COLS
        cols = GENE_QUANT_COLS
    elif datatype == "psms":
        if not all(x in df.columns.values for x in set(cols) - set(_EXTRA_COLS)):
            print("Potential error, not all columns filled.")
            print([x for x in cols if x not in df.columns.values])
        cols = [x for x in cols if x in df.columns.values]

    df.to_csv(
        os.path.join(outdir, outf),
        columns=cols,
        index=False,
        encoding="utf-8",
        sep="\t",
    )

    logging.info("Export of {} file : {}".format(labeltype, outf))


def assign_sra(df):
    # df['SRA'] = 'A'
    # cat_type = CategoricalDtype(categories=['S', 'R', 'A'],
    #                             ordered=True)
    # df['SRA'] = df['SRA'].astype(cat_type)
    df["SRA"] = pd.Categorical(
        ["A"] * len(df), categories=["S", "R", "A"], ordered=True
    )
    # df['SRA'] = df['SRA'].astype('category', categories=['S', 'R', 'A'],
    #                              ordered=True)
    df.loc[(df["IDSet"] == 1) & (df["IDGroup_u2g"] <= 3), "SRA"] = "S"

    df.loc[(df["IDSet"] == 2) & (df["IDGroup"] <= 3), "SRA"] = "S"

    df.loc[
        (df["IDSet"] == 1) & (df["SRA"] != "S") & (df["IDGroup_u2g"] <= 5), "SRA"
    ] = "R"

    df.loc[(df["IDSet"] == 2) & (df["SRA"] != "S") & (df["IDGroup"] <= 5), "SRA"] = "R"

    return df


# def set_protein_groups_axis(row, psms, protgis, protrefs):
# gid = row['GeneID']
# try:
#     gene_specific = protgis.get(gid).split(SEP)
# except AttributeError:
#     gene_specific = tuple()
# psms[ psms['GeneID'] == gid ]


def set_protein_groups_axis(row, protgis, protrefs):
    gid = row.name
    valid_protgis = protgis.get(gid, "").split(SEP)
    valid_refs = protrefs.get(gid, "").split(SEP)
    # protgis = filter(lambda x: not np.isnan(x), row.loc['ProteinGIs_All'])
    protgis = [x for x in row.loc["ProteinGIs_All"] if isinstance(x, str)]
    protrefs = [x for x in row.loc["ProteinRefs_All"] if isinstance(x, str)]

    # filter by proteins that map to this particular gene
    protgi_grps = [x for x in protgis if any(y in x for y in valid_protgis)]
    protref_grps = [x for x in protrefs if any(y in x for y in valid_refs)]

    # remove gis/refs that belong to different gids
    protgi_grps_gene, protref_grps_gene = list(), list()

    protgi_grps_gene = [
        set([x for x in grp.split(SEP) if x in valid_protgis]) for grp in protgi_grps
    ]
    # now find subsets:
    protgi_grps_pruned = list()
    for grp in protgi_grps_gene:
        for other in protgi_grps_gene:
            if grp == other:
                continue  # same one
            # if grp.intersection(other) == other:  # other is a subset
            #     break
            if not grp - other:  # grp is a subset
                break
        else:  # only if we get through everyother and don't break out do we keep
            if grp not in protgi_grps_pruned:
                protgi_grps_pruned.append(grp)

    protref_grps_gene = [
        set([x for x in grp.split(SEP) if x in valid_protgis]) for grp in protref_grps
    ]
    # now find subsets:
    protref_grps_pruned = list()
    for grp in protref_grps_gene:
        for other in protref_grps_gene:
            if grp == other:
                continue  # same one
            if grp.intersection(other) == other:  # other is a subset
                break
        else:  # only if we get through everyother and don't break out do we keep
            if grp not in protref_grps_pruned:
                protref_grps_pruned.append(grp)

    return "|".join(SEP.join(x) for x in protgi_grps_pruned), "|".join(
        SEP.join(x) for x in protref_grps_pruned
    )


def set_protein_groups(df, psms, gene_protgi_dict, gene_protref_dict):
    """Assign protein groups (via collections of unique-to-protein(s) peptides)"""

    res = (
        psms.groupby("GeneID")
        .agg({"ProteinGIs_All": "unique", "ProteinRefs_All": "unique"})
        .pipe(
            apply_by_multiprocessing,
            set_protein_groups_axis,
            func_args=(gene_protgi_dict, gene_protref_dict),
            workers=WORKERS,
            axis=1,
        )
        .apply(pd.Series)
        .rename(columns={0: "ProteinGI_GIDGroups", 1: "ProteinRef_GIDGroups"})
    )
    # this is regular expression count, so escape the pipe
    res["ProteinGI_GIDGroupCount"] = res.ProteinGI_GIDGroups.str.count("\|") + 1
    res["ProteinRef_GIDGroupCount"] = res.ProteinRef_GIDGroups.str.count("\|") + 1

    return df.merge(res, left_on="GeneID", right_index=True)


def assign_labels_for_qual(df, labeltype, labeltypes):
    if labeltype in ("none" "TMT"):
        df["LabelFLAG"] = ";".join(
            map(str, [labelflag.get(x, "?") for x in labeltypes])
        )
    else:
        raise NotImplementedError("")
    return df


# from ._orig_code import *
# from ._orig_code import (extract_peptideinfo, _extract_peptideinfo,
#                          _peptidome_matcher, peptidome_matcher)
def grouper(
    usrdata,
    outdir="",
    database=None,
    gid_ignore_file="",
    labels=None,
    contaminant_label="__CONTAMINANT__",
    razor=False,
):
    usrdata.df['is_decoy'] = False
    usrdata.df.loc[ usrdata.df.raw_headers_All.str.contains("rev_"), 'is_decoy' ] = True
    if labels is None:
        labels = dict()
    """Function to group a psm file from PD after Mascot Search"""

    def print_log_msg(df_or_None=None, msg="", *args, **kwargs):
        """Print and log a message.
        Can be used in pipe operation as always returns the
        first argument untouched.
        """
        fmt = dict(now=datetime.now(), msg=msg)
        log_msg = "{now} | {msg}".format(**fmt)
        logging.info(log_msg)
        return df_or_None

    # import RefseqInfo

    # file with entry of gene ids to ignore for normalizations
    gid_ignore_list = []
    usrdata_out = usrdata.output_name("psms", ext="tsv")
    if gid_ignore_file is not None and os.path.isfile(gid_ignore_file):
        print(f"Using gene filter file {gid_ignore_file}.")
        gid_ignore_list = get_gid_ignore_list(gid_ignore_file)

    area_col = "PrecursorArea"  # set default
    normalize = 10**9

    logging.info(
        f"Starting Grouper for exp file {usrdata.datafile}".format(usrdata.datafile)
    )
    logging.info("")
    logging.info(f"Filter values set to : {usrdata.filterstamp}")

    # ==================== Populate gene info ================================ #
    # gene_metadata = extract_metadata(usrdata.df.metadatainfo)

    # gene_taxon_dict = gene_taxon_mapper(database)
    gene_taxon_dict = database[["geneid", "taxon"]].set_index("geneid")["taxon"].to_dict()

    # gene_symbol_dict = gene_symbol_mapper(database)
    gene_symbol_dict = database[["geneid", "symbol"]].set_index("geneid")["symbol"].to_dict()

    # gene_desc_dict = gene_desc_mapper(database)
    gene_desc_dict = database[["geneid", "description"]].set_index("geneid")["description"].to_dict()

    gene_hid_dict = gene_hid_mapper(database)
    # gene_hid_dict = database[["geneid", "hid"]].set_index("geneid")["hid"].to_dict()

    # gene_protgi_dict = gene_protgi_mapper(database)
    gene_protgi_dict = database[["geneid", "gi"]].set_index("geneid")["gi"].to_dict()

    # gene_protref_dict = gene_protref_mapper(database)
    gene_protref_dict = database[["geneid", "ref"]].set_index("geneid")["ref"].to_dict()

    # ==================== Populate gene info ================================ #

    logging.info(f"Finished matching PSMs to taxonid: {usrdata.taxonid}")

    nomatches = usrdata.df[usrdata.df["GeneIDCount_All"] == 0].Sequence  # select all
    # PSMs that didn't get a match to the refseq peptidome
    matched_psms = len(usrdata.df[usrdata.df["GeneIDCount_All"] != 0])
    unmatched_psms = len(nomatches)  # these lack geneid in the fasta file header
    # generally contaminants / fasta file entries that
    logging.info(f"Total matched PSMs : {matched_psms}")
    logging.info(f"Total unmatched PSMs : {unmatched_psms}")

    # store this somewhere else?
    # for missing_seq in nomatches:
    #     logging.warning(
    #         "No match for sequence {} in {}".format(missing_seq, usrdata.datafile)
    #     )
    #     # Store all of these sequences in the big log file, not per experiment.
    # export the PSM quantification data to separate files.
    # store resulting files in `psm_data` pd.Series

    labeltypes = get_labels(usrdata.df, labels, usrdata.labeltype)

    if len(labeltypes) == 0:
        raise ValueError(f"No labels found indata for dtype {usrdata.labeltype}")
    psm_data = split_multiplexed(usrdata, labeltypes, exp_labeltype=usrdata.labeltype, tmt_reference=tmt_reference)
    # dictionary of labels : dataframe of data

    dtypes = usrdata.df.dtypes.to_dict()
    label_taxon = defaultdict(dict)  # store this for species estimates for each label

    qual_data = (
        usrdata.df.pipe(assign_IDG, filtervalues=usrdata.filtervalues)
        .assign(sequence_lower=lambda x: x["Sequence"].str.lower())
        .sort_values(
            by=[
                "SpectrumFile",
                area_col,
                "Sequence",
                "Modifications",
                "Charge",
                "PSM_IDG",
                "IonScore",
                "PEP",
                "q_value",
            ],
            ascending=[0, 0, 1, 1, 1, 1, 0, 1, 1],
        )
        .pipe(redundant_peaks)  # remove ambiguous peaks
    )

    # genes_df = (create_df(temp_df, label)
    good_qual_data = (
        qual_data.assign(PrecursorArea_split=lambda x: x["PrecursorArea"])
        .pipe(sum_area)  # we have to calculate this here, but will recalculate
        .pipe(auc_reflagger)  # for each label
        .pipe(
            flag_AUC_PSM,
            usrdata.filtervalues,
            contaminant_label=contaminant_label,
            phospho=usrdata.phospho,
        )
        .pipe(split_on_geneid)
        .assign(
            TaxonID=lambda x: x["GeneID"].map(gene_taxon_dict),
            Symbol=lambda x: x["GeneID"].map(gene_symbol_dict),
            Description=lambda x: x["GeneID"].map(gene_desc_dict),
            ProteinGIs=lambda x: x["GeneID"].map(gene_protgi_dict),
            ProteinRefs=lambda x: x["GeneID"].map(gene_protref_dict),
        )
        .pipe(select_good_peptides)
        .pipe(assign_labels_for_qual, usrdata.labeltype, labeltypes)
    )
    logging.info(f"Good qual data : {len(good_qual_data)}")

    psms_qual_f = os.path.join(
        usrdata.outdir, usrdata.output_name("psms_QUAL", ext="tsv")
    )
    # out = os.path.join(usrdata.outdir, usrdata.output_name('psms', ext='tab'))
    _cols = [
        x
        for x in DATA_ID_COLS
        if x in good_qual_data and x not in ("sequence_lower", "Sequence_set")
    ]
    _miss_cols = [x for x in DATA_ID_COLS if x not in good_qual_data]
    # if _miss_cols:
    #     logging.warning(
    #         "Missing 1 or more PSM QUAL columns: {}".format(", ".join(_miss_cols))
    #     )
    # good_qual_data.to_csv(out, index=False, encoding='utf-8', sep='\t', columns=DATA_ID_COLS)

    _toexclude = (
        "Sequence_set",
        "sequence_lower",
        "CreationTS",
        "metadatainfo",
        "Dataset",
        "index",
        "StatMomentsDataCountUsed",
        "PeakKSStat",
        "rawfile",  # extra column not used
        # *[
        #     x for x in good_qual_data if not re.match(r".*SignalToNoise$", x)
        # ],  # these are columns from MASIC
        # *[
        #     x for x in good_qual_data if not re.match(r".*Resolution$", x)
        # ],  # these are columns from MASIC
    )

    try:
        good_qual_data.to_csv(
            psms_qual_f,
            columns=[x for x in good_qual_data if x not in _toexclude],
            index=False,
            encoding="utf-8",
            sep="\t",
        )
    except Exception as e:
        print(e) # this could break things, prevent creation of psms_all concat file, but maybe not

    logging.info(f"Wrote {psms_qual_f}")

    qual_genes = (
        create_df(good_qual_data)
        # .assign(TaxonID = lambda x : x['GeneID'].map(gene_taxon_dict))
        .pipe(get_gene_info, database)
        # .pipe(get_gene_capacity, database)
        .pipe(get_peptides_for_gene, good_qual_data)
        .pipe(calc_coverage, database, good_qual_data)
        .pipe(get_psms_for_gene, good_qual_data)
        # .pipe(calculate_full_areas, temp_df, 'SequenceArea', normalize)
        # .fillna(0)
        .pipe(
            print_log_msg,
            msg="Assigning gene sets and groups for {}".format(usrdata.datafile),
        )
        .pipe(assign_gene_sets, good_qual_data)
        .pipe(set_gene_gpgroups, good_qual_data)
        .assign(
            Symbol=lambda x: x["GeneID"].map(gene_symbol_dict),
            Description=lambda x: x["GeneID"].map(gene_desc_dict),
            HIDs=lambda x: x["GeneID"].map(gene_hid_dict),
            ProteinGIs=lambda x: x["GeneID"].map(gene_protgi_dict),
            ProteinRefs=lambda x: x["GeneID"].map(gene_protref_dict),
            EXPRecNo=usrdata.recno,
            EXPRunNo=usrdata.runno,
            EXPSearchNo=usrdata.searchno,
            AddedBy=usrdata.added_by,
        )
        .pipe(assign_sra)
        .pipe(set_protein_groups, good_qual_data, gene_protgi_dict, gene_protref_dict)
        .pipe(assign_labels_for_qual, usrdata.labeltype, labeltypes)
    )

    if "PeptideSet" in qual_genes:
        qual_genes = qual_genes.drop(labels=["PeptideSet"], axis=1)
    genedata_out = usrdata.output_name("e2g_QUAL", ext="tsv")
    _outf = os.path.join(usrdata.outdir, genedata_out)

    qual_genes.to_csv(
        _outf,
        columns=GENE_QUAL_COLS,
        index=False,
        encoding="utf-8",
        sep="\t",
    )
    logging.info(f"Wrote {genedata_out}")
    # usrdata.e2g_files.append(_outf) # don't need to keep track of this one

    _id_cols = [
        x
        for x in [
            "SpectrumFile",
            "Charge",
            "RTmin",
            "FirstScan",
            "PSMAmbiguity",
            "ProteinList",
            "IonScore",
            "q_value",
            "SequenceModi",
            "Sequence",
            "EXPRecNo",
            "EXPRunNo",
            "EXPSearchNo",
            "DeltaMassPPM",
            "PEP",
            "MissedCleavages",
            "SequenceModiCount",
            "Modifications",
            "GeneIDs_All",
            "GeneIDCount_All",
            "TaxonIDs_All",
            "TaxonIDCount_All",
            "ProteinGIs_All",
            "ProteinGICount_All",
            "ProteinRefs_All",
            "ProteinRefCount_All",
            "HIDs",
            "HIDCount_All",
            "GeneCapacities",
            "PSM_IDG",
            "sequence_lower",
            "Peak_UseFLAG",
            "is_decoy",
        ]
        if x in qual_data
    ]

    gene_level_label_dict = dict()
    # data_cols = DATA_COLS
    # import ipdb; ipdb.set_trace()
    for label, df in psm_data.items():
        # label = flaglabel.get(labelix, 'none')
        # df = pd.read_table(psmfile, dtype=dtypes)
        # df = pd.read_table(psmfile)
        # df = psmfile
        # df["SpectrumFile"] = df["SpectrumFile"].fillna("")

        logging.info(f"Working on label {label}")
        if df.empty:
            logging.warning(f"No data for {label}")
            continue

        # for some reason the RTmin values get off and need to be rounded to match
        # have to do this to get rid of floating point decimal inaccuracy..
        # this is done because the data file is reloaded from disk
        # if this is all kept in memory we could avoid this
        # df['RTmin'] = df.RTmin.mul(10e6).pipe(np.floor).div(10e6).round(5)
        # df['RTmin'] = df.RTmin.mul(10e2).pipe(np.floor).div(10e2).round(5)
        # qual_data['RTmin'] = qual_data['RTmin'].mul(10e2).pipe(np.floor).div(10e2).round(5)
        # df["RTmin"] = df.RTmin.pipe(np.around, 4)
        # qual_data["RTmin"] = qual_data["RTmin"].pipe(np.around, 4)

        _merge_cols = [
            x
            for x in [
                "EXPRecNo",
                "EXPRunNo",
                "EXPSearchNo",
                "FirstScan",  # have to have this
                "SpectrumFile",
                "Charge",
                "SequenceModi",
            ]
        ]
        #                if x in _id_cols
        # ]

        dfm = df.merge(
            qual_data[_id_cols],
            on=_merge_cols,
            indicator=True,
            how="outer",
        )

        assert dfm._merge.value_counts().both <= len(dfm)

        dfm = (
            dfm.pipe(sum_area)
            .pipe(auc_reflagger)  # remove duplicate sequence areas
            .pipe(
                flag_AUC_PSM,
                usrdata.filtervalues,
                contaminant_label=contaminant_label,
                phospho=usrdata.phospho,
            )
            .pipe(split_on_geneid)
            .assign(
                TaxonID=lambda x: x["GeneID"].map(gene_taxon_dict),
                Symbol=lambda x: x["GeneID"].map(gene_symbol_dict),
                Description=lambda x: x["GeneID"].map(gene_desc_dict),
                # ProteinGIs=lambda x: x["GeneID"].map(gene_protgi_dict),
                ProteinRefs=lambda x: x["GeneID"].map(gene_protref_dict),
                LabelFLAG=label,
            )
        )

        additional_labels = list()
        # ======================== Plugin for multiple taxons  ===================== #
        # taxon_ids = usrdata.df['TaxonID'].replace(['0', 0, 'NaN', 'nan', 'NAN'], np.nan).dropna().unique()
        taxon_ids = (
            #dfm["TaxonID"]
            dfm[ dfm.GeneID != contaminant_label ]["TaxonID"]
            .replace(["0", 0, "NaN", "nan", "NAN", -1, "-1", ""], np.nan)
            .dropna()
            .unique()
        )

        # import ipdb; ipdb.set_trace()
        taxon_totals = dict()
        # usrdata.to_logq("TaxonIDs: {}".format(len(taxon_ids)))
        logging.info("TaxonIDs: {}".format(len(taxon_ids)))
        # usrdata.to_logq(str(usrdata.df))
        if len(taxon_ids) == 1 or usrdata.no_taxa_redistrib:  # just 1 taxon id present
            for tid in taxon_ids:  # taxon_ids is a set
                taxon_totals[tid] = 1
                label_taxon[label][tid] = 1
        elif len(taxon_ids) > 1:  # more than 1 taxon id
            taxon_totals = multi_taxon_splitter(
                taxon_ids, dfm, gid_ignore_list, "PrecursorArea_split"
            )
            logging.info("Multiple taxa found, redistributing areas...")
            usrdata.taxon_ratio_totals.update(taxon_totals)
            for taxon, ratio in taxon_totals.items():
                # print('For the full data : {} = {}'.format(taxon, ratio))
                # usrdata.to_logq('For the full data : {} = {}'.format(taxon, ratio))
                logging.info("For label {} : {} = {}".format(label, taxon, ratio))
                # usrdata.to_logq("For label {} : {} = {}".format(label, taxon, ratio))
                label_taxon[label][taxon] = ratio

        (gpgcount, genecount, ibaqtot,) = (
            0,
            0,
            0,
        )

        orig_area_col = "SequenceArea"
        # Don't use the aggregated SequenceArea for TMT experiments
        logging.info(f"Starting genen assignment for label {label}")

        # if usrdata.labeltype in ('TMT', 'iTRAQ'):
        #     isobar_output = pd.DataFrame()  # instead of merging, we will concat
        # for label in labeltypes:  # increase the range to go through more label types
        #     labelix = labelflag.get(label, 0)
        #     area_col = orig_area_col
        # usrdata.to_logq('{} | Starting gene assignment for label {}.'.format(
        #     time.ctime(), label))

        # dfm['TaxonID'].fillna('', inplace=True)
        # dfm['Symbol'].fillna('', inplace=True)
        # dfm['Description'].fillna('', inplace=True)
        # dfm['ProteinRefs'].fillna('', inplace=True)
        # dfm['SequenceModiCount'].fillna('', inplace=True)
        temp_df = select_good_peptides(dfm)

        if temp_df.empty:  # only do if we actually have peptides selected
            logging.warning("No good peptides found for label {}".format(label))
            continue

        # if usrdata.labeltype in ('TMT', 'iTRAQ'):
        #     # isobar_area_col = 'PrecursorArea'  # we always use Precursor Area
        #     # temp_df, area_col = redistribute_area_isobar(temp_df, label,
        #     #                                              labeltypes,
        #     #                                              area_col=isobar_area_col,
        #     #                                              labeltype=usrdata.labeltype)
        #     if len(taxon_ids) > 1 and not usrdata.no_taxa_redistrib:  # more than 1 taxon id
        #         print('Calculating taxon ratios for label {}'.format(label))
        #         usrdata.to_logq('{} | Calculating taxon ratios for label{}.'.format(time.ctime(), label))
        #         taxon_totals = multi_taxon_splitter(taxon_ids, temp_df,
        #                                             gid_ignore_list, area_col='PrecursorArea_split')
        #         for taxon, ratio in taxon_totals.items():
        #             print('For label {} : {} = {}'.format(label, taxon, ratio))
        #             usrdata.to_logq('For label {} : {} = {}'.format(label, taxon, ratio))
        # elif usrdata.labeltype == 'SILAC':
        #     raise NotImplementedError('No support for SILAC experiments yet.')
        # ==================================================================== #

        logging.info(
            "Populating gene table for {}.".format(datetime.now(), usrdata.datafile)
        )

        # msg_areas = "Calculating peak areas for {}".format(usrdata.datafile)
        # msg_sets = "Assigning gene sets and groups for {}".format(usrdata.datafile)

        genes_df = (
            create_df(temp_df)
            .merge(qual_genes, on="GeneID")
            # .pipe(get_gene_info, database)
            # .pipe(get_peptides_for_gene, temp_df)
            # .pipe(calc_coverage, database, temp_df)
            # .pipe(get_psms_for_gene, temp_df)
            # .pipe(print_log_msg, msg=msg_areas)
            .pipe(calculate_full_areas, temp_df, "SequenceArea", normalize)
            .assign(LabelFLAG=label)
            # .pipe(print_log_msg, msg=msg_sets)
            # .pipe(assign_gene_sets, temp_df)
            # .pipe(set_gene_gpgroups, temp_df)
            # .assign(Symbol = lambda x: x['GeneID'].map(gene_symbol_dict),
            #         Description = lambda x: x['GeneID'].map(gene_desc_dict),
            #         HIDs = lambda x: x['GeneID'].map(gene_hid_dict),
            #         ProteinGIs = lambda x: x['GeneID'].map(gene_protgi_dict),
            #         ProteinRefs = lambda x: x['GeneID'].map(gene_protref_dict),
            # )
            # .pipe(assign_sra)
            # .pipe(set_protein_groups, temp_df, gene_protgi_dict, gene_protref_dict)
        )

        msg_areas = "Calculating distributed area ratio for {}".format(usrdata.datafile)
        logging.info(msg_areas)
        # print_log_msg(df=None, msg=msg_areas)
        distribute_area(
            temp_df,
            genes_df,
            "SequenceArea",
            taxon_totals,
            not usrdata.no_taxa_redistrib,
        )
        now = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        genes_df['GeneCapacity'] = genes_df.GeneCapacity.apply(lambda x: max(x, 1))
        genes_df = (
            genes_df.pipe(calculate_gene_dstrarea, temp_df, normalize)
            # .pipe(calculate_gene_razorarea, temp_df, normalize)
            .assign(
                iBAQ_dstrAdj=lambda x: np.divide(
                    x["AreaSum_dstrAdj"], x["GeneCapacity"]
                )
            )
            .sort_values(by=["GPGroup"], ascending=True)
            .reset_index()
            # .assign(EXPRecNo = pd.Categorical(usrdata.recno),
            #         EXPRunNo = pd.Categorical(usrdata.runno),
            #         EXPSearchNo = pd.Categorical(usrdata.searchno),
            #         AddedBy = pd.Categorical(usrdata.added_by),
            #         CreationTS = pd.Categorical(now),
            #         ModificationTS = pd.Categorical(now))
            # .assign(EXPRecNo = usrdata.recno,
            #         EXPRunNo = usrdata.runno,
            #         EXPSearchNo = usrdata.searchno,
            #         AddedBy = usrdata.added_by,
            #         CreationTS = now,
            #         ModificationTS = now)
            .sort_values(by=["IDSet", "GPGroup"])
        )
        if razor:
            genes_df = genes_df.pipe(calculate_gene_razorarea, temp_df, normalize)

        # =============================================================#

        genecount += len(genes_df)
        ibaqtot += genes_df[~genes_df.GeneID.isin(gid_ignore_list)].iBAQ_dstrAdj.sum()

        if usrdata.labeltype in ("TMT", "iTRAQ", "SILAC"):  # and silac
            genedata_out = usrdata.output_name(
                regularize_filename(label) + "_e2g", ext="tsv"
            )
        else:
            genedata_out = usrdata.output_name("e2g_QUANT", ext="tsv")
        if razor and "RazorArea" not in E2G_COLS:
            E2G_COLS.append("AreaSum_razor")
        # genes_df.to_csv(os.path.join(usrdata.outdir, genedata_out), columns=E2G_COLS,

        _outf = os.path.join(usrdata.outdir, genedata_out)
        genes_df.to_csv(
            _outf,
            columns=GENE_QUANT_COLS,
            index=False,
            encoding="utf-8",
            sep="\t",
        )
        usrdata.e2g_files.append(_outf)

        # genedata_chksum = os.path.join(usrdata.outdir, usrdata.output_name('e2g', ext='md5'))
        # write_md5(genedata_chksum, md5sum(os.path.join(usrdata.outdir, genedata_out)))

        # write_md5(genedata_out_chksum, md5sum(os.path.join(usrdata.outdir, genedata_out)))
        del genes_df

        logging.info(f"Export of genetable for labeltype {label} completed.".format())

        # if len(usrdata.df) == 0:
        #     print('No protein information for {}.\n'.format(repr(usrdata)))
        #     usrdata.to_logq('No protein information for {}.\n'.format(repr(usrdata)))
        #     usrdata.flush_log()
        #     return

        # usrdata.df['HIDs'] = ''  # will be populated later
        # usrdata.df['HIDCount'] = ''  # will be populated later

        logging.info("Starting peptide ranking.")
        # dstr_area = 'PrecursorArea_dstrAdj'
        # area_col_to_use = dstr_area if dstr_area in usrdata.df.columns else orig_area_col
        area_col_to_use = "PrecursorArea_dstrAdj"

        newcols = list(set(temp_df.columns) - set(dfm.columns))
        dfm = dfm.join(temp_df[newcols])
        # usrdata.df = (pd.merge(usrdata.df, temp_df, how='left')
        # usrdata.df = (usrdata.df.join(temp_df[newcols])
        # ranked = (dfm.join(temp_df[newcols])[['GeneID', area_col_to_use, 'SequenceModi', 'Charge',
        ranked = (
            dfm[
                [
                    "GeneID",
                    area_col_to_use,
                    "SequenceModi",
                    "Charge",
                    "PSM_IDG",
                    "IonScore",
                    "PEP",
                    "q_value",
                    "Modifications",
                    "AUC_UseFLAG",
                    "PSM_UseFLAG",
                    "Peak_UseFLAG",
                    "LabelFLAG",
                ]
            ].pipe(rank_peptides, area_col="PrecursorArea_dstrAdj", ranks_only=True)
            # .assign(PrecursorArea_split = lambda x: x['PrecursorArea'],
            # .assign(PeptRank = lambda x: (x['PeptRank']
            #                               .fillna(0)
            #                               .astype(np.integer))
            # )
        )
        ranked = ranked.fillna(0).astype(int).to_frame("PeptRank")
        dfm = dfm.join(ranked)

        # usrdata.df['ModificationTS'] = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        # usrdata.df['PrecursorArea_split'] = usrdata.df['PrecursorArea']
        # import psutil; print('\n', psutil.virtual_memory())
        # didn't get a rank gets a rank of 0
        msg = "Peptide ranking complete for {}.".format(usrdata.datafile)
        logging.info(msg)

        _outf = os.path.join(
            usrdata.outdir,
            usrdata.output_name(
                suffix=regularize_filename(label) + "_psms_QUANT", ext="tsv"
            ),
        )

        Q_COLS = [x for x in DATA_QUANT_COLS if x in dfm]
        not_Q_COLS = [x for x in DATA_QUANT_COLS if x not in dfm]

        dfm.to_csv(_outf, index=False, encoding="utf-8", sep="\t", columns=Q_COLS)
        usrdata.psm_files.append(_outf)

        msfname = usrdata.output_name("msf", ext="tsv")
        # msfname = usrdata.output_name('msf', ext='tab')
        msfdata = spectra_summary(usrdata, dfm)
        _outf = os.path.join(usrdata.outdir, msfname)
        msfdata.to_csv(_outf, index=False, sep="\t")
        logging.info(f"Wrote {_outf}")
        # usrdata.msf_files.append(_outf)

        # usrdata.df = None

    export_metadata(
        program_title=program_title,
        usrdata=usrdata,
        matched_psms=matched_psms,
        unmatched_psms=unmatched_psms,
        usrfile=usrdata.datafile,
        taxon_totals=usrdata.taxon_ratio_totals,
        outname=usrdata.output_name("metadata", ext="json"),
        outpath=usrdata.outdir,
    )

    # we don't need any of this anymore
    if usrdata.labeltype in ("TMT", "iTRAQ", "SILAC"):
        # TODO fix this to concat dict of dataframes instead of loading files
        # OR NOT?

        #  concat_isobar_output(
        #      usrdata.recno,
        #      usrdata.runno,
        #      usrdata.searchno,
        #      usrdata.outdir,
        #      usrdata.output_name(suffix="e2g_QUANT"),
        #      labeltype=usrdata.labeltype,
        #      datatype="e2g",
        #  )

        _files = [x for x in usrdata.e2g_files if "QUAL" not in x]
        # if not _files: continue
        _df = pd.concat([pd.read_table(f) for f in _files])
        _outf = os.path.join(usrdata.outdir, usrdata.output_name(suffix="e2g_QUANT"))
        _overlapping_cols = [x for x in GENE_QUANT_COLS if x in _df]
        _df.to_csv(_outf, columns=_overlapping_cols, index=False, sep="\t")
        logging.info(f"Wrote {_outf}")

        data_cols = DATA_QUANT_COLS
        if razor and "RazorArea" not in data_cols:
            data_cols.append("RazorArea")

        _df = pd.concat(
            [pd.read_table(x) for x in usrdata.psm_files if "QUAL" not in x]
        )
        _outf = os.path.join(usrdata.outdir, usrdata.output_name(suffix="psms_QUANT"))
        _overlapping_cols = [x for x in data_cols if x in _df]
        _df.to_csv(_outf, index=False, columns=_overlapping_cols, sep="\t")
        logging.info(f"Wrote {_outf}")

        # usrdata.clean()

    # import ipdb; ipdb.set_trace()
    # usrdata.df = pd.merge(usrdata.df, temp_df, how='left')
    data_cols = DATA_QUANT_COLS
    concat_isobar_output(
        usrdata.recno,
        usrdata.runno,
        usrdata.searchno,
        usrdata.outdir,
        usrdata.output_name(suffix="psms_QUANT"),
        labeltype=usrdata.labeltype,
        datatype="psms",
        cols=data_cols,
    )

    # import ipdb; ipdb.set_trace()
    # usrdata.clean()

    concat_isobar_output(
        usrdata.recno,
        usrdata.runno,
        usrdata.searchno,
        usrdata.outdir,
        usrdata.output_name(suffix="msf"),
        labeltype=usrdata.labeltype,
        datatype="msf",
    )

    # isobar_output.to_csv(os.path.join(usrdata.outdir, usrdata_out), columns=data_cols,
    #                      index=False, encoding='utf-8', sep='\t')
    # else:
    #     if not all(x in usrdata.df.columns.values for x in set(data_cols) - set(_EXTRA_COLS)):
    #         print('Potential error, not all columns filled.')
    #         print([x for x in data_cols if x not in usrdata.df.columns.values])
    #     data_cols = [x for x in data_cols if x in usrdata.df.columns.values]
    #     usrdata.df.to_csv(os.path.join(usrdata.outdir, usrdata_out), columns=data_cols,
    #                       index=False, encoding='utf-8', sep='\t')

    out = os.path.join(usrdata.outdir, usrdata.output_name("species_ratios", ext="tsv"))
    label_taxonratios = pd.DataFrame(label_taxon)
    label_taxonratios.to_csv(out, index=True, encoding="utf-8", sep="\t")

    # usrdata.to_logq(
    #     "{} | Export of datatable completed.".format(time.ctime())
    #     + "\nSuccessful grouping of file completed."
    # )
    # usrdata.flush_log()

    print("Successful grouping of {} completed.\n".format(repr(usrdata)))

    return


def calculate_breakup_size(row_number, enzyme="trypsin"):
    # print(32)
    # print(ceil(row_number/32))
    # return ceil(row_number/32)
    if enzyme == "noenzyme":
        return ceil(row_number / 140)
    return ceil(row_number / 4)


def set_modifications(usrdata):
    to_replace = {
        "DeStreak": "des",
        "Deamidated": "dam",
        "Carbamidomethyl": "car",
        "Oxidation": "oxi",
        "Phospho": "pho",
        "Acetyl": "ace",
        "GlyGly": "gg",
        "Label:13C(6)": "lab",
        "Label:13C(6)+GlyGly": "labgg",
        "\)\(": ":",
        "79.9663": "pho",
        "229.1629": "TMT6",
        "304.207": "TMT16",
        "57.0215": "car",
        "15.9949": "oxi",
        # "": "",
    }
    modis_abbrev = usrdata.Modifications.fillna("").replace(regex=to_replace).fillna("")
    modis_abbrev.name = "Modifications_abbrev"
    usrdata = usrdata.join(modis_abbrev)

    # labels = usrdata.Modifications.str.extract('(Label\S+)(?=\))').unique()  # for SILAC
    no_count = "tmt", "itraq", "Label", "label"

    modifications = usrdata.apply(
        lambda x: seq_modi(
            x["Sequence"],
            x["Modifications_abbrev"],
            no_count,
            # labels=labels
        ),
        axis=1,
    )
    (
        usrdata["Sequence"],
        usrdata["SequenceModi"],
        usrdata["SequenceModiCount"],
        usrdata["LabelFLAG"],
    ) = zip(*modifications)
    return usrdata


def load_fasta(refseq_file):
    REQUIRED_COLS = ("geneid", "sequence")
    ADDITIONAL_COLS = ("description", "gi", "homologene", "ref", "taxon", "symbol")

    gen = fasta_dict_from_file(refseq_file, header_search="specific")
    df = pd.DataFrame.from_dict(gen)  # dtype is already string via RefProtDB

    # gen2 = fasta_dict_from_file(refseq_file, header_search="generic")
    # df2 = pd.DataFrame.from_dict(gen2)  # dtype is already string via RefProtDB

    # if not all(x in df.columns for x in REQUIRED_COLS):
    #     missing = ", ".join(x for x in REQUIRED_COLS if x not in df.columns)
    #     fmt = "Invalid FASTA file : {} is missing the following identifiers : {}\n"
    #     raise ValueError(fmt.format(refseq_file, missing))
    missing_cols = (x for x in ADDITIONAL_COLS if x not in df.columns)
    for col in missing_cols:
        df[col] = ""
    return df


ENZYME = {
    "trypsin": dict(cutsites=("K", "R"), exceptions=("P",)),
    "trypsin/P": dict(cutsites=("K", "R"), exceptions=None),
    "chymotrypsin": dict(cutsites=("Y", "W", "F", "L"), exceptions=None),
    "LysC": dict(cutsites=("K",), exceptions=None),
    "GluC": dict(cutsites=("E",), exceptions=None),
    "ArgC": dict(cutsites=("R",), exceptions=None),
    "noenzyme": dict(cutsites=()),
}
# TODO: add the rest
# 'LysN', 'ArgC'


def _match(
    usrdatas,
    refseq_file,
    miscuts=2,
    enzyme="trypsin/P",
    semi_tryptic=False,
    semi_tryptic_iter=6,
    min_pept_len=7,
):
    # expand later
    enzyme_rule = parser.expasy_rules.get(enzyme, "noenzyme")
    if enzyme == "noenzyme":
        enzyme_rule = "."
        miscuts = 50
    print("Using peptidome {} with rule {}".format(refseq_file, enzyme))

    # database = pd.read_table(refseq_file, dtype=str)
    # rename_refseq_cols(database, refseq_file)
    database = load_fasta(refseq_file)
    database["capacity"] = np.nan
    breakup_size = calculate_breakup_size(len(database), enzyme=enzyme)
    print(f"breakup size {breakup_size}")
    counter = 0
    prot = defaultdict(list)
    # for ix, row in tqdm.tqdm(database.iterrows(), total=len(database)):
    for ix, row in database.iterrows():
        counter += 1
        fragments, fraglen = protease(
            row.sequence,
            minlen=min_pept_len,
            # cutsites=['K', 'R'],
            # exceptions=['P'],
            rule=enzyme_rule,
            miscuts=miscuts,
            # semi = False if enzyme_rule != 'noenzyme' else True,
            semi_tryptic=semi_tryptic,  # not functional
            semi_tryptic_iter=semi_tryptic_iter,
            # **enzyme_rule,
        )
        database.loc[ix, "capacity"] = fraglen
        for (
            fragment
        ) in fragments:  # store location in the DataFrame for the peptide's parent
            prot[fragment].append(ix)

        if counter > breakup_size:
            for usrdata in usrdatas:
                usrdata.df = peptidome_matcher(
                    usrdata.df, prot
                )  # match peptides to peptidome
            counter = 0
            del prot  # frees up memory, can get quite large otherwise
            prot = defaultdict(list)
    else:
        for usrdata in usrdatas:
            usrdata.df = peptidome_matcher(
                usrdata.df, prot
            )  # match peptides to peptidome
        del prot

    # now extract info based on index
    for usrdata in usrdatas:
        if usrdata.searchdb is None:
            usrdata.searchdb = refseq_file
        extract_peptideinfo(usrdata, database)

    return database


def match(
    usrdatas,
    refseqs,
    enzyme="trypsin/P",
    semi_tryptic=False,
    min_pept_len=7,
    semi_tryptic_iter=6,
    miscuts=2,
):
    """
    Match psms with fasta database
    Input is list of UserData objects and an optional dictionary of refseqs
    """
    inputdata_refseqs = set([usrdata.taxonid for usrdata in usrdatas])
    databases = dict()
    sortfunc = lambda x: x.taxon_miscut_id
    usrdata_sorted = sorted(usrdatas, key=sortfunc)
    for k, g in itertools.groupby(usrdata_sorted, key=sortfunc):
        group = list(g)
        taxonid = group[0].taxonid
        miscuts = group[0].miscuts
        refseq = refseqs.get(taxonid, refseqs.get(""))

        if refseq is None:
            err = "No refseq file available for {}".format(taxonid)
            warn(err)
            for u in group:
                u.ERROR = err
                u.EXIT_CODE = 1
            continue

        database = _match(
            group,
            refseq,
            miscuts=miscuts,
            enzyme=enzyme,
            semi_tryptic=semi_tryptic,
            semi_tryptic_iter=semi_tryptic_iter,
            min_pept_len=min_pept_len,
        )
        databases[taxonid] = database
    # for organism in refseqs:
    #     if any(x == int(organism) for x in inputdata_refseqs):
    #                                     # check if we even need to read the
    #                                     # peptidome for that organism
    #         database = _match([usrdata for usrdata in usrdatas if usrdata.taxonid==organism],
    #                           refseqs[organism])
    #         databases[organism] = database

    return databases


def column_identifier(df, aliases):
    column_names = dict()
    for col in aliases:
        for alias in aliases[col]:
            name = [dfcolumn for dfcolumn in df.columns if dfcolumn == alias]
            if len(name) == 1:
                column_names[col] = name[0]
                break
    return column_names


# Idea
REQUIRED_HEADERS = [
    "Sequence",
    "Modifications",
    "PrecursorArea",
    "Charge",
    "IonScore",
    "q_value",
    "PEP",
    "SpectrumFile",
    "RTmin",
    "DeltaMassPPM",
]


def check_required_headers(df):
    if not all(x in df.columns for x in REQUIRED_HEADERS):
        missing = [x for x in REQUIRED_HEADERS if x not in df.columns]
        if "Modifications" in missing and "SequenceModi" in df:
            # then we are OK
            return
        fmt = "Invalid input file, missing {}".format(", ".join(missing))
        raise ValueError(fmt)


def set_up(usrdatas, column_aliases, enzyme="trypsin/P", protein_column=None):
    """Set up the usrdata class for analysis
    Read data, rename columns (if appropriate), populate base data"""
    if not usrdatas:
        return
    for usrdata in usrdatas:
        EXIT_CODE = usrdata.read_csv(
            sep="\t",
        )  # read from the stored psms file
        if EXIT_CODE != 0:
            logging.error("Error with reading {!r}".format(usrdata))
            continue

        if column_aliases:
            standard_name_mapper = column_identifier(usrdata.df, column_aliases)
            protected_names = ["Modified sequence", "SequenceModi"]
            if protein_column:
                protected_names.append(protein_column)

            rev_mapping = {v: k for k, v in standard_name_mapper.items()}

            if protein_column is not None:
                protein_column_out = rev_mapping.get(protein_column, protein_column)
            else:
                protein_column_out = None

            usrdata.df.rename(
                columns=rev_mapping,
                inplace=True,
            )
            redundant_cols = [
                x
                for x in usrdata.df.columns
                if (x not in standard_name_mapper.keys() and x not in protected_names)
            ]
            # usrdata.df = usrdata.df.drop(redundant_cols, axis=1)

        # usrdata.df = usrdata.populate_base_data()
        # some data exports include positional info about sequences
        # such as: [K].xxxxxxxxxxxK.[N]
        # we want to remove that and only have the exact peptide sequence
        usrdata.df["Sequence"] = usrdata.df.Sequence.str.extract(
            "(\w{3,})", expand=False
        )
        usrdata.populate_base_data()
        if "DeltaMassPPM" not in usrdata.df:
            usrdata.df["DeltaMassPPM"] = 0
        if "SpectrumFile" not in usrdata.df:
            usrdata.df["SpectrumFile"] = ""
        if "RTmin" not in usrdata.df:
            usrdata.df["RTmin"] = 0
        if "PEP" not in usrdata.df.columns:
            usrdata.df["PEP"] = 0  # not trivial to make a category due to sorting
            # usrdata.categorical_assign('PEP', 0, ordered=True)
        if not "q_value" in usrdata.df.columns:
            logging.warning("No q_value available")
            usrdata.df["q_value"] = 0
        if "PrecursorArea" not in usrdata.df:
            usrdata.df["PrecursorArea"] = 1
            # try:
            #     usrdata.df["q_value"] = usrdata.df["PEP"] / 10  # rough approximation
            # except KeyError:
            #     warn("")
            #     usrdata.df["q_value"] = 0

        check_required_headers(usrdata.df)

        # targets, exceptions = ENZYME[enzyme]['cutsites'], ENZYME[enzyme]['exceptions']
        # usrdata.df.apply(lambda x: calculate_miscuts(x['Sequence'], targets=targets,
        #                                               exceptions=exceptions ),
        #                  axis=1)

        if "MissedCleavages" not in usrdata.df.columns:
            targets, exceptions = (
                ENZYME[enzyme]["cutsites"],
                ENZYME[enzyme]["exceptions"],
            )
            usrdata.df["MissedCleavages"] = usrdata.df.apply(
                lambda x: calculate_miscuts(
                    x["Sequence"], targets=targets, exceptions=exceptions
                ),
                axis=1,
            )
        if not "PSMAmbiguity" in usrdata.df.columns:
            # usrdata.df['PSMAmbiguity'] = 'Unambiguous'
            usrdata.categorical_assign("PSMAmbiguity", "Umambiguous")  # okay?

        # ===================================================================

        if not usrdata.pipeline == "MQ" and not any(
            x in usrdata.df for x in ("Modified sequence", "SequenceModi")
        ):  # MaxQuant already has modifications
            usrdata.df = set_modifications(usrdata.df)
        else:
            # usrdata.df['SequenceModi'] = usrdata.df['Modified sequence']
            usrdata.df.rename(
                columns={"Modified sequence": "SequenceModi"}, inplace=True
            )
            if "Modifications" in usrdata.df:
                usrdata.df["SequenceModiCount"] = count_modis_maxquant(
                    usrdata.df, usrdata.labeltype
                )
            elif "SequenceModi" in usrdata.df:  # should be if we get to this point
                usrdata.df["SequenceModiCount"] = count_modis_seqmodi(
                    usrdata.df, usrdata.labeltype
                )
                # usrdata.categorical_assign("Modifications", "")
                usrdata.df["Modifications"] = ""

        if usrdata.df.SequenceModi.isna().any():
            usrdata.df.loc[
                usrdata.df.SequenceModi.isna(), "SequenceModi"
            ] = usrdata.df.loc[usrdata.df.SequenceModi.isna(), "Sequence"]
            # usrdata.df['SequenceModi'] = usrdata.df.apply(lambda x: x['Sequence'] if pd.isna(x['SequenceModi'])
            #                                               else x['SequenceModi'], 1)

            usrdata.categorical_assign("LabelFLAG", 0)  # TODO: handle this properly
    return protein_column_out


def rename_refseq_cols(df, filename):
    fasta_h = ["TaxonID", "HomologeneID", "GeneID", "ProteinGI", "FASTA"]
    # regxs = {x: re.compile(x) for x in _fasta_h}
    to_rename = dict()
    for header in fasta_h:
        matched_col = [col for col in df.columns if header in col]
        if len(matched_col) != 1:
            raise ValueError(
                """Could not identify the correct
            column in the reference sequence database for '{}'
            for file : {}""".format(
                    header, filename
                )
            )
        to_rename[matched_col[0]] = header
    df.rename(columns=to_rename, inplace=True)


WORKERS = 1


def main(
    usrdatas=[],
    fullpeptread=False,
    inputdir="",
    outputdir="",
    refs=dict(),
    rawfilepath=None,
    column_aliases=dict(),
    gid_ignore_file="",
    labels=dict(),
    raise_on_error=False,
    contaminant_label="__CONTAMINANT__",
    enzyme="trypsin/P",
    semi_tryptic=False,
    semi_tryptic_iter=6,
    min_pept_len=7,
    tmt_reference=None,
    workers=1,
    razor=False,
    miscuts=2,
    protein_column=None,
    protein_columntype=None,
):
    """
    refs :: dict of taxonIDs -> refseq file names
    """
    # ====================Configuration Setup / Loading======================= #
    global WORKERS
    WORKERS = workers
    if imagetitle:
        fancyprint(program_title, 12)  # ascii rt
        #  fancyprint('Malovannaya lab',10)  #
    elif not imagetitle:
        print(program_title)  # not as exciting as ascii art

    logging.info("release date: {}".format(__copyright__))
    logging.info("Python version " + sys.version)
    logging.info("Pandas version: " + pd.__version__)
    logging.info("=" * 79)

    startTime = datetime.now()

    # print("\nStart at {}".format(startTime))
    logging.info("Start at {}".format(startTime))

    # first set the modifications. Importantly fills in X with the predicted amino acid
    protein_column = set_up(usrdatas, column_aliases, enzyme, protein_column)
    # for ud in usrdatas:
    #     print(ud, ud.EXIT_CODE)
    if all(usrdata.EXIT_CODE != 0 for usrdata in usrdatas):
        return usrdatas

    # special case where we want to skip matching to database and just use pre-mapped column
    if protein_column is not None:
        databases = dict()

        for usrdata in usrdatas:
            database = map_to_gene(
                usrdata, protein_column, identifier=protein_columntype
            )

            databases[usrdata.taxonid] = database

    else:
        databases = match(
            [x for x in usrdatas if x.EXIT_CODE == 0],
            refs,
            enzyme=enzyme,
            semi_tryptic=semi_tryptic,
            semi_tryptic_iter=semi_tryptic_iter,
            min_pept_len=min_pept_len,
            miscuts=miscuts,
        )

    if all(usrdata.EXIT_CODE != 0 for usrdata in usrdatas):
        return usrdatas

    # failed_exps = []
    for usrdata in usrdatas:
        if usrdata.EXIT_CODE != 0:
            continue
        try:
            grouper(
                usrdata,
                database=databases[usrdata.taxonid],
                gid_ignore_file=gid_ignore_file,
                labels=labels,
                contaminant_label=contaminant_label,
                razor=razor,
            )
            usrdata.EXIT_CODE = 0
        except Exception as e:  # catch and store all exceptions, won't crash
            # the whole program at least
            usrdata.EXIT_CODE = 1
            usrdata.ERROR = traceback.format_exc()
            stack = traceback.format_exc()
            # failed_exps.append((usrdata, e))
            # usrdata.to_logq(
            #     "Failure for file of experiment {}.\n"
            #     "The reason is : {}".format(repr(usrdata), e)
            # )
            # usrdata.to_logq(stack)
            logging.error(stack)
            # for s in stack:
            #     usrdata.to_logq(s, sep='')
            #     print(s, sep='')
            # usrdata.flush_log()
            logging.error("Failure for file of experiment {}.".format(repr(usrdata)))
            logging.error("Traceback:\n {}".format(e))

            if raise_on_error:
                raise  # usually don't need to raise, will kill the script. Re-enable
                # if need to debug and find where errors are
    logging.info("Time taken : {}\n".format(datetime.now() - startTime))
    return usrdatas
