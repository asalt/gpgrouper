from __future__ import print_function

import os
import sys
import re
import shutil
import six

if six.PY3:
    from itertools import zip_longest
    from configparser import ConfigParser
elif six.PY2:
    from itertools import izip_longest as zip_longest
    from ConfigParser import ConfigParser
from getpass import getuser

from pathlib import Path
from datetime import datetime
import multiprocessing
import warnings

import click

from . import subfuncts, gpgrouper, _version
from .split_diann_export import split_diann_export
from .containers import UserData
from .parse_config import parse_configfile, find_configfile, Config


__author__ = "Alexander B. Saltzman"
__copyright__ = "Copyright January 2016"
__credits__ = ["Alexander B. Saltzman", "Anna Malovannaya"]
__license__ = "MIT"
__version__ = _version.__version__
__maintainer__ = "Alexander B. Saltzman"
__email__ = "saltzman@bcm.edu"


CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])
CONFIG_NAME = "gpgrouper_config.ini"
BASE_CONFIG = os.path.join(os.path.dirname(os.path.realpath(__file__)), CONFIG_NAME)


class Config(object):
    def __init__(self, user):
        self.user = user
        self.ispec_url = None
        self.database = None
        self.outfile = "-"
        self.filtervalues = dict()
        self.column_aliases = dict()
        self.CONFIG_DIR = "."
        self.inputdir = "."
        self.outputdir = "."
        self.rawfiledir = "."
        self.labels = dict()
        self.refseqs = dict()
        self.fastadb = None
        self.contaminants = None


if six.PY3:
    parser = ConfigParser(
        comment_prefixes=(";")
    )  # allow number sign to be read in configfile
elif six.PY2:
    parser = ConfigParser()  # allow number sign to be read in configfile
parser.optionxform = str


@click.group(name="main")
@click.version_option(__version__)
@click.pass_context
def cli(ctx):
    pass


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.option("-p", "--path", type=click.Path(exists=True), default=".")
def getconfig(path):
    """Generate a new config file
    (and also parent directory if necessary)"""
    shutil.copy(BASE_CONFIG, path)
    click.echo("Created new config file {} at {}".format(CONFIG_NAME, path))


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.argument("path", type=click.Path(exists=True), default=".", required=False)
def openconfig(path):
    config_file = find_configfile(path=path)
    if config_file is None:
        click.echo("Could not find config file in {}".format(path))
        return
    click.launch(config_file)


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.option("-s", "--search-no", default=4, show_default=True, type=int)
@click.argument("diann-export", type=click.Path(exists=True))
def prepare_diann(search_no, diann_export):
    print(diann_export)
    split_diann_export(diann_export, search_no=search_no)


def validate_cores(ctx, param, value):

    cores = multiprocessing.cpu_count()

    if value < 1 or value > cores:
        raise click.BadParameter("must be between 1 and {}".format(cores))

    return value


DEFAULTS = {
    "max_files": 99,
    "pep": 1.0,
    "enzyme": "trypsin/P",
    "configfile": None,
    # "interval": 3600,
    "rawfiledir": ".",
    "taxonid": None,
    "contaminants": None,
    "quant_source": "AUC",
    "outdir": None,
    "zmax": 6,
    "ion_score": 7.0,
    "ion_score_bins": (10.0, 20.0, 30.0),
    "qvalue": 0.05,
    "idg": 9,
    "autorun": False,
    "name": "shiro",
    "modi": 4,
    "zmin": 2,
    "no_taxa_redistrib": False,
    "labeltype": "none",
    "pipeline": "PD",
    "miscuts": 2,
}


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "-c",
    "--contaminants",
    type=click.Path(exists=True, dir_okay=False),
    help="Contaminants file of IDs to ignore when calculating with multiple taxa",
)
@click.option(
    "--contaminant-label",
    default="__CONTAMINANT__",
    show_default=True,
    help="Contaminant label within FASTA file",
)
@click.option(
    "-d",
    "--database",
    type=click.Path(exists=True, dir_okay=False),
    help="Database file to use. Ignored with autorun.",
)
@click.option(
    "-e",
    "--enzyme",
    type=click.Choice(
        [
            "trypsin",
            "trypsin/P",
            "chymotrypsin",
            "lysc",
            # "LysN",
            "arg-c" "asp-n" "cnbr",
            "noenzyme",
            # "GluC",
            # "ArgC" "AspN",
        ]
    ),
    default=DEFAULTS["enzyme"],
    show_default=True,
    help="Enzyme used for digestion. Ignored with autorun.",
)
@click.option(
    "--ion-score",
    type=float,
    default=DEFAULTS["ion_score"],
    show_default=True,
    help="Ion score cutoff for a psm.",
)
@click.option(
    "--ion-score-bins",
    nargs=3,
    type=float,
    default=DEFAULTS["ion_score_bins"],
    show_default=True,
    help="The three (ascending) IonScore cutoffs used to place PSMs in quality bins.",
)
@click.option(
    "-l",
    "--labeltype",
    type=click.Choice(["none", "SILAC", "iTRAQ", "TMT"]),
    default=DEFAULTS["labeltype"],
    show_default=True,
    help="Type of label for this experiment.",
)
@click.option(
    "--tmt-reference",
    help="reference channel to ratio all other channels by when calculating psm auc. only used if --labeltype == TMT",
)
@click.option("--min-pept-len", default=7, show_default=True)
@click.option(
    "-m",
    "--miscuts",
    type=int,
    default=DEFAULTS["miscuts"],
    help="Number of allowed miscuts",
    show_default=True,
)
@click.option(
    "--modi",
    type=int,
    default=DEFAULTS["modi"],
    show_default=True,
    help="Maximum modifications to allow on one peptide",
)
@click.option(
    "-n",
    "--name",
    type=str,
    default=getuser(),
    show_default=True,
    help="Name associated with the search",
)
@click.option(
    "-ntr",
    "--no-taxa-redistrib",
    is_flag=True,
    show_default=True,
    help="Disable redistribution based on individual taxons",
)
@click.option(
    "-o",
    "--outdir",
    type=click.Path(file_okay=False),
    default=None,
    help="Output directory for files.",
    show_default=True,
)
@click.option(
    "-p",
    "--psms-file",
    type=click.Path(exists=True, dir_okay=False),
    multiple=True,
    help="Tab deliminated file of psms to be grouped",
)
@click.option(
    "--phospho",
    is_flag=True,
    default=False,
    show_default=True,
    help="Only use phospho-modified peptides for quantification",
)
@click.option(
    "--pipeline",
    type=click.Choice(["PD", "MQ"]),
    show_default=True,
    help="Pipeline used for generating PSM file.",
)
@click.option(
    "--idg",
    type=int,
    default=DEFAULTS["idg"],
    show_default=True,
    help="PSM IDG cutoff value.",
)
@click.option(
    "--pep",
    type=float,
    default=DEFAULTS["pep"],
    show_default=True,
    help="Posterior error probability cutoff",
)
@click.option(
    "--qvalue",
    type=float,
    default=DEFAULTS["qvalue"],
    show_default=True,
    help="Cutoff q-value for a given psm.",
)
@click.option(
    "-r",
    "--rawfiledir",
    type=click.Path(file_okay=False),
    default=DEFAULTS["rawfiledir"],
    show_default=True,
    help="""Directory to look for corresponding rawfiles for extra summary information.
              Note the raw files are not required for successful analysis.""",
)
@click.option(
    "--razor",
    default=False,
    show_default=True,
    is_flag=True,
    help="""Also calculate Peptide and Gene Area based on Razor Peptide
              definition. Left here for comparison.""",
)
@click.option(
    "-s",
    "--configfile",
    type=click.Path(exists=True, dir_okay=False),
    help="""'Points to a specific configfile for use in the analysis.
              Note will automatically look for a `gpgrouper_config.ini` in present directory if not specified""",
)
@click.option(
    "--semi-tryptic",
    is_flag=True,
    default=False,
    show_default=True,
    help="""(experimental) Semi-tryptic enzymatic digestion. Right now only goes a max of 3 AAs in each direction
              for performance considerations.""",
)
@click.option("--semi-tryptic-iter", default=6, type=int, show_default=True)
@click.option("-t", "--taxonid", type=str, help="(experimental) semi-tryptic-iter")
@click.option(
    "--protein-column",
    default=None,
    type=str,
    show_default=True,
    help="""
              protein identifier column to use. Will attempt to map protein accession numbers to pre-annotated
              protein gene mapping table using NCBI protein gi or accession.
              Each entry is a semicolon (;) delimited list of protein accession numbers.
              Assumes all peptides valid matches and ignores fasta database and miscuts flag.
              Use in combination with `--protein-columntype`
              """,
)
@click.option("--protein-columntype", type=click.Choice(["ref", "gi"]))
@click.option(
    "--zmin",
    type=int,
    default=DEFAULTS["zmin"],
    show_default=True,
    help="Minimum charge",
)
@click.option(
    "--zmax",
    type=int,
    default=DEFAULTS["zmax"],
    show_default=True,
    help="Maximum charge",
)
@click.option(
    "--record-no",
    type=int,
    help="record numbers for the corresponding PSM files",
    multiple=True,
)
@click.option(
    "--run-no",
    type=int,
    multiple=True,
    help="run numbers for the corresponding PSM files",
)
@click.option(
    "--search-no",
    type=int,
    multiple=True,
    help="search numbers for the corresponding PSM files",
)
@click.option(
    "--workers",
    type=int,
    callback=validate_cores,
    default=1,
    show_default=True,
    help="""Number of workers to use when multiprocessing is appropriate.
              Has no effect if the OS cannot fork""",
)
def run(
    contaminants,
    contaminant_label,
    database,
    enzyme,
    ion_score,
    ion_score_bins,
    labeltype,
    min_pept_len,
    miscuts,
    modi,
    name,
    no_taxa_redistrib,
    outdir,
    psms_file,
    pipeline,
    idg,
    pep,
    qvalue,
    rawfiledir,
    configfile,
    semi_tryptic,
    semi_tryptic_iter,
    taxonid,
    tmt_reference,
    zmin,
    zmax,
    protein_column,
    protein_columntype,
    phospho,
    record_no,
    run_no,
    search_no,
    workers,
    razor,
):
    """Run gpGrouper"""

    if not all([database, psms_file]) and (not database and not protein_column):
        click.echo("No database or psms file entered, showing help and exiting...")
        click.echo(click.get_current_context().get_help())
        sys.exit(1)

    if configfile is None:
        configfile = BASE_CONFIG
    config = parse_configfile(
        configfile
    )  # will parse if config file is specified or gpgrouper_config.ini exists in PD
    if config is None:
        config = Config(name)
    INPUT_DIR = config.inputdir
    OUTPUT_DIR = outdir or config.outputdir or "."
    RAWFILE_DIR = rawfiledir or config.rawfiledir
    LABELS = config.labels
    refseqs = config.refseqs
    filtervalues = None  # just get from cmd line
    # if filtervalues:
    # s = ', '.join(['ion_score', 'ion_score_bins', 'qvalue', 'pep', 'idg', 'zmin', 'zmax', 'modi'])
    # s = ", ".join(["ion_score", "qvalue", "pep", "idg", "zmin", "zmax", "modi"])
    # click.echo(
    #     "Using predefined config file. Specifying any of {} via a flag will have no effect".format(
    #         s
    #     )
    # )
    column_aliases = config.column_aliases
    gid_ignore_file = contaminants or config.contaminants

    usrdatas = list()
    if not taxonid:
        taxonid = ""
        # taxonid = click.prompt(
        #     "Enter taxon id",
        #     default=9606,
        #     type=int,
        # )
    for ix, psmfile in enumerate(psms_file):
        if record_no:
            recno = record_no[ix]
            try:
                runno = run_no[ix]
            except IndexError:
                runno = 1
            try:
                searchno = search_no[ix]
            except IndexError:
                searchno = 1
        else:
            res = find_rec_run_search(psmfile)
            if res:
                recno, runno, searchno = res
            else:  # regex search failed, just use a default
                recno, runno, searchno = None, None, None
                # recno = ix + 1  # default recno starts at 1
                # runno, searchno = 1, 1
        usrdata = UserData(
            recno=recno,
            runno=runno,
            searchno=searchno,
            taxonid=taxonid,
            datafile=psmfile,
            indir=INPUT_DIR,
            outdir=OUTPUT_DIR,
            rawfiledir=RAWFILE_DIR,
            no_taxa_redistrib=no_taxa_redistrib,
            labeltype=labeltype,
            addedby=name,
            phospho=phospho,
            searchdb=database,
            miscuts=miscuts,
        )
        refseqs[taxonid] = database
        # INPUT_DIR, usrfile = os.path.split(Path(psmfile).resolve().__str__())
        INPUT_DIR, usrfile = os.path.split(os.path.abspath(psmfile))
        usrdata.indir, usrdata.datafile = INPUT_DIR, usrfile
        usrdata.outdir = OUTPUT_DIR or Path(OUTPUT_DIR).resolve().__str__()
        # later on expected that datafile is separated from path
        usrdata.quant_source = "AUC"
        usrdata.pipeline = pipeline
        if filtervalues:  # if defined earlier from passed config file
            usrdata.filtervalues = filtervalues
            params = click.get_current_context().params
            for param in (
                "ion_score",
                "ion_score_bins",
                "qvalue",
                "pep",
                "idg",
                "zmin",
                "zmax",
                "modi",
            ):
                # need to check for explictly passed options
                if params[param] != DEFAULTS[param]:  # Explicitly over-write
                    usrdata.filtervalues[param] = params[param]
        else:
            usrdata.filtervalues["ion_score"] = ion_score
            usrdata.filtervalues["ion_score_bins"] = ion_score_bins
            usrdata.filtervalues["qvalue"] = qvalue
            usrdata.filtervalues["pep"] = pep
            usrdata.filtervalues["idg"] = int(idg)
            usrdata.filtervalues["zmin"] = int(zmin)
            usrdata.filtervalues["zmax"] = int(zmax)
            usrdata.filtervalues["modi"] = int(modi)

        usrdatas.append(usrdata)

    ret = gpgrouper.main(
        usrdatas=usrdatas,
        inputdir=INPUT_DIR,
        outputdir=OUTPUT_DIR,
        refs=refseqs,
        column_aliases=column_aliases,
        gid_ignore_file=contaminants,
        labels=LABELS,
        tmt_reference=tmt_reference,  # only used if labeltype == TMT
        contaminant_label=contaminant_label,
        enzyme=enzyme,
        semi_tryptic=semi_tryptic,
        semi_tryptic_iter=semi_tryptic_iter,
        min_pept_len=min_pept_len,
        workers=workers,
        razor=razor,
        miscuts=miscuts,
        protein_column=protein_column,
        protein_columntype=protein_columntype,
    )
    if not all(x.EXIT_CODE == 0 for x in ret):
        for x in ret:
            x.flush_log()

        raise RuntimeError("\n".join([x.ERROR for x in ret]))


def find_rec_run_search(target):
    "Try to get record, run, and search numbers with regex of a target string with pattern \d+_\d+_\d+"

    _, target = os.path.split(target)  # ensure just searching on filename
    rec_run_search = re.compile(r"^(\d+)_(\d+)_(\d+)_")

    match = rec_run_search.search(target)
    if match:
        recno, runno, searchno = match.groups()
        return recno, runno, searchno
    return

    # match = rec_run_search.search(target).group()
    # recno = re.search(r'^\d+', match).group()
    # recno_pat = re.compile('(?<={}_)\d+'.format(recno))
    # runno = re.search(recno_pat, match).group()
    # runno_pat = re.compile('(?<={}_{}_)\d+'.format(recno, runno))
    # searchno = re.search(runno_pat, match).group()
    # return recno, runno, searchno


if __name__ == "__main__":
    pass
    # from click.testing import CliRunner
    # test()
