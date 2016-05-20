import sys
import os
import shutil
import argparse
from itertools import repeat
import multiprocessing as mp
from pygrouper import pygrouper
#sys.path.append('..')  # access to pygrouper
#import pygrouper


#print('Initializing pygrouper testing...')
#if not os.path.isdir('./testresults'):
#    print('Removing previous test results.')
#    shutil.rmtree('./testresults')
#    os.mkdir('./testresults')

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def grab_one():
    setups = [{'EXPRecNo': 30595,
               'EXPRunNo': 1,
               'EXPSearchNo': 2,
               'taxonID': 9606,
               'EXPQuantSource': 'AUC',
               'AddedBy': 'test',
               'EXPTechRepNo': 1,
               'EXPLabelType': 'none'
    },
    ]
    files = ['30595_1_2_3T3_LM2_5_5_PROF_75MIN_all.txt']
    files = [os.path.join(BASE_DIR, f) for f in files]
    return (files, setups)

def grab_data(quick, prof, tmt):

    setups = [{'EXPRecNo': 30490,
               'EXPRunNo': 1,
               'EXPSearchNo': 1,
               'taxonID': 9606,
               'EXPQuantSource': 'AUC',
               'AddedBy': 'test',
               'EXPTechRepNo': 1,
               'EXPLabelType': 'none'
                     },
              {'EXPRecNo': 30404,
               'EXPRunNo': 1,
               'EXPSearchNo': 1,
               'taxonID': 20150811,
               'EXPQuantSource': 'AUC',
               'AddedBy': 'test',
               'EXPTechRepNo': 1,
               'EXPLabelType': 'none'
                     },
              {'EXPRecNo': 30259,
               'EXPRunNo': 1,
               'EXPSearchNo': 1,
               'taxonID': 20150811,
               'EXPQuantSource': 'AUC',
               'AddedBy': 'test',
               'EXPTechRepNo': 1,
               'EXPLabelType': 'none'
                     },
              ]
    if quick:
        files = ['30490_1_EQP_6KiP_all.txt']
    elif tmt:
        files = ['30490_1_EQP_6KiP_all_tmt.txt']
        setups[0]['EXPLabelType'] = 'TMT'
    elif prof:
        files = ['30404_1_QEP_ML262_75min_020_RPall.txt']
        setups = [setups[1]]
    else:
        files = ['30490_1_EQP_6KiP_all.txt','30404_1_QEP_ML262_75min_020_RPall.txt']



    inputdir = None

    return (files, setups, inputdir)

def make_processes(max_processes, data_args):

    processes = []
    more = True
    while len(processes) <= max_processes:
        try:
            inputs = next(data_args)
            processes.append(mp.Process(target=pygrouper.main, args=inputs))
        except StopIteration:
            more = False

    return (processes, more)


def runtest(quick=False, prof=False, tmt=False, testone=False, **kwargs):
    inputdir = BASE_DIR
    files, setups, inputdir = grab_data(quick, prof, tmt)
    if testone: # overwrite previous
        files, setups =  grab_one()
    if inputdir is None:
        inputdir = os.getcwd()
    kwargs['outputdir'] = './testresults'
    print('Files for testing :')
    files = [os.path.join(BASE_DIR, f) for f in files]
    for f in files:
        print(f)
    print('running test')
    pygrouper.main(usrfiles=files, exp_setups=setups,
                   automated=True,
                   usedb=False, **kwargs)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="""Script to test pygrouper""",

        epilog=""" """
        )

    parser.add_argument('-q','--quick', action='store_true',
                        help='Run one small file.')
    parser.add_argument('-p','--profile', action='store_true',
                        help='Run one large file.')
    parser.add_argument('-t', '--test', action='store_true',
                        help='Run 1 specific test file.')

    args = parser.parse_args()
    runtest(quick=args.quick, prof=args.profile, testone=args.test)
