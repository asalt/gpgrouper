import os
import re
import pandas as pd
from functools import partial
from pathlib import Path

def get_reader(file):
    if file.endswith("tsv"):
        return partial(pd.read_csv, sep='\t')
    if file.endswith("parquet"):
        return pd.read_parquet

def validate_df(df):
    """
    Validate that the DataFrame has the required columns.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    
    Raises:
    AssertionError: If 'File.Name' is not in the DataFrame columns
    """
    if "File.Name" not in df.columns:
        if "Run" in df.columns:
            df["File.Name"] = df["Run"]
        else:
            raise ValueError("The DataFrame must contain a 'File.Name' column.")
    if "Global.Q.Value" not in df.columns:
        raise ValueError("The DataFrame must contain a 'Global.Q.Value' column.")
    return df

#def split_diann_export(file, searchno = 4):
#    df = pd.read_table(file)
#    validate_df(df)
#    # df['samplegroup'] = df['File.Name'].str.extract(r'(\d{5}_\d+)')
#    df['basefilename'] = df['File.Name'].apply(lambda x: os.path.basename(x.replace('\\', '/')))
#    df['rec_run'] = df['basefilename'].str.extract(r'^(\d{5}_\d+)')
#
#    SEARCHNO=searchno
#    # g = df.groupby('File.Name')
#    g = df.groupby('rec_run')
#    for grp_name, grp in g:
#        #outf = ix.split('\\')[-1].strip('.raw')+".tsv"
#        outf = f"{grp_name}_{SEARCHNO}_psms_all.tsv"
#        print(f"Writing {outf}")
#        grp.to_csv(outf, sep='\t', index=False)

def check_for_channels(df):
    if "Channel" not in df.columns or df['Channel'].nunique() == 1:
        return None
    valid_channels = df[ df['Channel'] != ""]['Channel'].unique()
    return valid_channels


OUTPUT_PATTERN="{rec_run}_{search_no}{channel}_psms_all.tsv" # channel will append a _ if present

def split_diann_export(file, search_no=4, output_pattern=OUTPUT_PATTERN):
    """
    Process and split a DIANN export file by 'rec_run' identifier and save each group to a separate file.
    
    Parameters:
    file (str): Path to the DIANN export file
    search_no (int): Search number to append to output files
    output_pattern (str): Naming pattern for output files, should include '{rec_run}' and '{search_no}'
    
    Returns:
    None
    """
    # Load data
    reader = get_reader(file)
    df = reader(file)
    # df = pd.read_table(file)
    df = validate_df(df)

    # Clean up 'File.Name' column for base filename extraction
    df['basefilename'] = df['File.Name'].apply(lambda x: Path(x.replace('\\', '/')).name)
    # Extract the rec_run identifier from basefilename
    df['rec_run'] = df['basefilename'].str.extract(r'^(\d{5}_\d+)')[0]
    # If 'rec_run' wasn't found at all, we may end up with all NaNs.
    if df['rec_run'].isna().all():
        print("No valid 'rec_run' identifiers found. No files will be created.")
        return

    # Group by 'rec_run' and write each group to a separate file
    channels = check_for_channels(df)
    if channels is not None:
        df = df[ df["Channel"].isin(channels) ]

    group_by = ['rec_run']
    if channels is not None:
        group_by.append("Channel")


    groups = df.groupby(group_by)


    for group_key, group in groups:
        _search_no = search_no
        if len(group_key) == 2:
            rec_run, channel = group_key
        else:
            rec_run = group_key[0]
            channel = None
        if pd.isna(rec_run):
            print("Skipping group with missing rec_run.")
            continue
        if channel is not None:
            channelstring = "_" + channel
            if channel == "H":
                _search_no += 1
        else:
            channelstring = ""


        if channel is not None:
            filtered_df = group[group['Channel.Q.Value'] < 0.01]
        else:
            filtered_df = group[group['Global.Q.Value'] < 0.01]


        # Format the output file name using the provided pattern

        outf = output_pattern.format(rec_run=rec_run,
                                     search_no=_search_no,
                                     channel=channelstring)
        print(f"Writing {outf}")
        group.to_csv(outf, sep='\t', index=False)

# Example usage:
# split_diann_export("path/to/diann_export_file.txt")

