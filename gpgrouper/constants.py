# constants.py

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
    "TMT_135": 1350,
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
    "RTmin",
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
