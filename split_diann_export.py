import os
import re
import pandas as pd


def validate_df(df):
    assert "File.Name" in df

def split_diann_export(file):
    df = pd.read_table(file)
    validate_df(df)
    # df['samplegroup'] = df['File.Name'].str.extract(r'(\d{5}_\d+)')
    df['basefilename'] = df['File.Name'].apply(lambda x: os.path.basename(x.replace('\\', '/')))
    df['rec_run'] = df['basefilename'].str.extract(r'^(\d{5}_\d+)')

    SEARCHNO=4
    # g = df.groupby('File.Name')
    g = df.groupby('rec_run')
    for grp_name, grp in g:
        #outf = ix.split('\\')[-1].strip('.raw')+".tsv"
        outf = f"{grp_name}_{SEARCHNO}_psms_all.tsv"
        print(f"Writing {outf}")
        grp.to_csv(outf, sep='\t', index=False)



