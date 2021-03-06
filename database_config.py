import os
from datetime import datetime
from collections import defaultdict
import numpy as np
import pandas as pd
from sqlalchemy import Column, ForeignKey, String, Integer, Float, \
Text, Date, Boolean, create_engine, MetaData, Table, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, backref
from sqlalchemy.orm.exc import NoResultFound, MultipleResultsFound



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Base = declarative_base()
def create_tables(engine):
    ''' Create a brand new database with tables'''
    Base.metadata.create_all(engine)

def get_connection():
    """Returns an sql connection with a cursor accessible"""
    dbname = os.path.join(BASE_DIR, 'piSPEC.sqlite3')
    engine = create_engine('sqlite:///{}'.format(dbname), echo=False)
    if not os.path.isfile(dbname):  # make db if doesn't exist
        create_tables(engine)
    return engine

def make_session():
    """Returns an SQLAlchemy Session object for the piSPEC.sqlite3 database"""
    dbname = os.path.join(BASE_DIR, 'piSPEC.sqlite3')
    engine = create_engine('sqlite:///{}'.format(dbname), echo=False)
    if not os.path.isfile(dbname):  # make db if doesn't exist
        create_tables(engine)

    Session = sessionmaker(bind=engine)
    session = Session()
    return session

def get_metadata():
    """Returns SQLAlchemy Metadata object for piSPEC.sqlite3 database"""
    dbname = os.path.join(BASE_DIR, 'piSPEC.sqlite3')
    engine = create_engine('sqlite:///{}'.format(dbname), echo=False)
    metadata = MetaData(bind=engine)
    return metadata

class Experiment(Base):

    __tablename__ = 'experiments'
    record_no = Column(Integer, primary_key=True)
    added_by = Column(String(100))
    creation_ts = Column(DateTime)
    exp_type = Column(String(50))

    runs = relationship('ExperimentRun',
                        order_by='ExperimentRun.id',
                        backref='experiments',
                        )

    def __repr__(self):
        return "<Experiment(record_no=%s>" % self.record_no

class ExperimentRun(Base):

    __tablename__ = 'experimentruns'
    id = Column(Integer, primary_key=True)

    record_no = Column(Integer, ForeignKey('experiments.record_no'))
    run_no = Column(Integer, default=1)
    search_no = Column(Integer, default=1)
    taxonid = Column(Integer)
    #added_by = Column(String(100), ForeignKey('experiments.added_by'))
    #creation_ts = Column(Date, ForeignKey('experiments.creation_ts'))
    purpose = Column(String(100))
    label_type = Column(String(30))
    tech_repeat = Column(Integer)
    MS_instrument = Column(String(100))
    MS_experimenter = Column(String(100))
    MS_comments = Column(String(100))
    quant_source = Column(String(30), default='AUC')
    search_experimenter = Column(String(100))

    #exp_type = Column(String(100))
    #identifier = Column(String(100))
    #project = Column(String(100))
    #primary_laboratory = Column(String(100))
    #exp_class = Column(String(100))  # eg : Affinity, Profiling
    #experimenter = Column(String(100))
    #exp_date = Column(Date)
    #description = Column(String(200))
    #conclusions = Column(Text)
    #cell_tissue = Column(String(200))
    #genotype = Column(String(200))
    #fractions = Column(String(200))
    #treatment = Column(String(200))
    #amount = Column(String(200))
    #affinity_amount = Column(String(200))
    #affinity_rec_no = Column(Integer)
    #protocol_no = Column(String(200))  # has letters too
    #washes = Column(String(200))
    #separation_1 = Column(String(200))
    #separation_1_detail = Column(String(200))
    #separation_experimenter = Column(String(200))
    digest_type = Column(String(100))
    digest_enzyme = Column(String(100), default='trypsin')
    #digest_experimenter = Column(String(200))
    PSM_count = Column(Integer)
    GPGroup_count = Column(Integer)
    gene_count = Column(Integer)
    iBAQ_total = Column(Float)
    group_date = Column(DateTime)
    grouped = Column(Boolean, default=False)
    notified = Column(Boolean, default=False)
    failed = Column(Boolean, default=False)

    exprec = relationship('Experiment',
                          backref=backref('experimentruns',
                          )
    )
    exp2gene = relationship('Gene',
                            backref='experimentruns',
                            )


    def __repr__(self):
        return ("<ExperimentRun(record_no={0}, run_no={1},"
                "search_no={2}, tech_repeat={3})>").format(self.record_no,
                                                           self.run_no,
                                                           self.search_no,
                                                           self.tech_repeat)


def add_exp2gene(df):
    ''' add pygrouper df to exp2gene table
    still in development, don't use yet
    '''
    rec = df[['e2g_EXPRecNo', 'e2g_EXPRunNo', 'e2g_GeneID', 'e2g_IDSet',
              'e2g_IDGroup', 'e2g_IDGroup_u2g', 'e2g_GPGroup', 'e2g_GPGroups_All',
              'e2g_PSMs', 'e2g_PSMs_u2g', 'e2g_PeptidePrint', 'e2g_PeptideCount',
              'e2g_PeptideCount_u2g', 'e2g_PeptideCount_S', 'e2g_PeptideCount_S_u2g',
              'e2g_nGPArea_Sum_cgpAdj', 'e2g_nGPArea_Sum_u2g', 'e2g_nGPArea_Sum_u2g_all',
              'e2g_nGPArea_Sum_max', 'e2g_nGPArea_Sum_dstrAdj', 'e2g_GeneCapacity',
              'e2g_n_iBAQ_dstrAdj']].to_records(index=False)
    genes = []
    session = make_session()
    #exprecord = session.query(db.ExperimentRun).\
    #                filter_by(record_no=exp_setup['EXPRecNo']).\
    #                filter_by(run_no=exp_setup['EXPRunNo']).one()
    for r in rec:
        #record = Experiment(record_no=rec[0][0])
        #run = ExperimentRun(run_no=rec[0][1])
        #record = rec[0][0]
        #run = rec[0][1]
        if isinstance(r[6], float) and not np.isnan(r[6]):
            gpg = int(r[6])
        else:
            gpg = None
        for x in [r[i] for i in range(2,12)]:
            if isinstance(x, np.int64):
                x = int(x)

        gene = Gene(#record_no=record,
            #run_no=run,
            run_id=1,
            GeneID=int(r[2]),
            IDSet=int(r[3]),
            IDGroup=r[4],
            IDGroup_u2g=r[5],
            GPGroup=gpg,
            GPGroups=r[7],
            PSMs=r[8],
            PSMs_u2g=r[9],
            Peptide_Print=r[10],
            Peptide_Count=r[11],
            Peptide_Count_u2g=r[12],
        )
        genes.append(gene)
    session.add_all(genes)
    session.commit()
    session.close()

class Gene(Base):

    __tablename__ = 'genes'
    id = Column(Integer, primary_key=True)
    #record_no = Column(Integer, ForeignKey('experiments.record_no'))
    #run_no = Column(Integer, ForeignKey('experimentruns.run_no'))
    run_id = Column(Integer, ForeignKey('experimentruns.id'))
    GeneID = Column(Integer)
    IDSet = Column(Integer)
    IDGroup = Column(Integer)
    IDGroup_u2g = Column(Integer)
    GPGroup = Column(Integer)
    GPGroups = Column(String(100))
    PSMs = Column(Integer)
    PSMs_u2g = Column(Integer)
    Peptide_Print = Column(Text)
    Peptide_Count = Column(Integer)
    Peptide_Count_u2g = Column(Integer)
    Peptide_Count_S = Column(Integer)
    Peptide_Count_S_u2g = Column(Integer)
    GPArea_Sum_cgpAdj = Column(Float)
    GPArea_Sum_u2g = Column(Float)
    GPArea_Sum_u2g_all = Column(Float)
    GPArea_Sum_max = Column(Float)
    GPArea_Sum_u2g = Column(Float)
    GPArea_Sum_dstrAdj = Column(Float)
    Gene_Capacity = Column(Float)
    iBAQ_dstrAdj = Column(Float)
    def __repr__(self):
        return ("<Gene(id={0}, record_no={1}, run_no={2}, GeneID={3}, "
                "IDSet={4}, iBAQ={5})>").format(self.id,
                                                self.id,
                                                self.run_id,
                                                self.GeneID,
                                                self.IDSet,
                                                self.iBAQ_dstrAdj)
    #exp2gene = relationship('Experiment', secondary='experimentruns',
    #                        primaryjoin='Gene.run_no==ExperimentRun.run_no',
    #                        secondaryjoin='Experiment.record_no==ExperimentRun.record_no')
    #experiment = relationship('Experiment', backref=backref('genes'))
    experimentrun = relationship('ExperimentRun', backref=backref('genes'))

class Peptide(Base):

    __tablename__ = 'peptide'
    id = Column(Integer, primary_key=True)

def get_ungrouped_exps():
    """Returns a list of all ungrouped experiments"""
    session = make_session()
    ungrouped = session.query(ExperimentRun).filter_by(grouped=False).filter_by(failed=False).all()
    session.close()
    return ungrouped

def get_grouped_exps():
    """Returns a list of all grouped experiments"""
    session = make_session()
    grouped = session.query(ExperimentRun).filter_by(grouped=True).filter_by(failed=False).all()
    session.close()
    return grouped
    

def update_exp_run(rec, run, passed=False, gpg=0, psms=0, ibaq=0):
    """Updates experiment run record."""
    session = make_session()
    try:
        q = session.query(ExperimentRun).join(
            Experiment).filter(
                Experiment.record_no==rec).filter(
                    ExperimentRun.run_no==run).one()
    except (NoResultFound, MultipleResultsFound) as e:
        print('Error updating experiment {}_{}').format(rec, run)
        return

    if passed:
        q.grouped = True
        q.GPGroup_count = gpg
        q.PSM_count = psms
        q.iBAQ_total = ibaq
    elif not passed:
        q.failed = True

    session.add(q)
    session.close()

def add_experiments(newexps):
    ''' newexps is a list of dictionaries'''
    session = make_session()
    expinfo = []

    for e in newexps:
        q = session.query(Experiment).filter(Experiment.record_no == e)
        expruninfo = []
        try:
            exp = q.one()
        except NoResultFound:
            
            exp = Experiment(record_no=e,
                             added_by=newexps[e][0].get('addedby'),
                             creation_ts=newexps[e][0].get('creation_ts'),
                             exp_type=newexps[e][0].get('exptype')
            )
        for rec in newexps[e]:
            exprun = ExperimentRun(
                #record_no = rec.get('rec_no'),
                run_no=rec.get('run_no'),
                search_no=rec.get('search_no'),
                taxonid=rec.get('taxon'),
                #added_by=rec.get('addedby'),
                #creation_ts=rec.get('creation_ts'),
                purpose=rec.get('purpose'),
                label_type=rec.get('label'),
                tech_repeat=rec.get('techrep'),
                MS_instrument=rec.get('instrument'),
                MS_experimenter=rec.get('msexperimenter'),
                quant_source=rec.get('quant'),
                search_experimenter=rec.get('searchexperimenter'),
                digest_type=rec.get('digest_type'),
                digest_enzyme=rec.get('digest_enzyme'),
            )
            if session.query(ExperimentRun).filter_by(record_no=e,
                                                      run_no=rec.get('run_no'),
                                                      search_no=rec.get('search_no'),
                                                      tech_repeat=rec.get('techrep')).count()==0:
                expruninfo.append(exprun)
        for run in expruninfo:
            exp.experimentruns.append(run)  # a list of ExperimentRun() entries

        expinfo.append(exp)

    session.add_all(expinfo)
    session.commit()
    session.close()

def get_all_experiment_records_slower():    

    exps = defaultdict(list)
    metadata = get_metadata()
    expruns = Table('experimentruns', metadata, autoload=True)
    s = expruns.select(expruns)
    rs = s.execute()
    for rec in rs:
        exps[rec.record_no].append(rec.run_no)
    rs.close()

    return exps

def get_all_experiment_records():

    session = make_session()
    exps = defaultdict(list)
    for exp in session.query(Experiment).all():
        rec = exp.record_no
        for exp1, run in session.query(Experiment, ExperimentRun).\
            filter(Experiment.record_no == rec).filter(ExperimentRun.record_no == rec).all():

            exps[rec].append(run.run_no)

    session.close()
    return exps


def get_ispec_experiment_info(ispecf, todb=False, norepeats=False):
    ''' get the ispec xlsx file with the experiment data and convert to dict'''
    print('Looking for new records')
    if norepeats:
        d = get_all_experiment_records()
    newexps = defaultdict(list)
    ispecdata = pd.read_excel(ispecf)
    intfields = ['exprun_EXPRecNo', 'exprun_EXPRunNo', 'exprun_EXPSearchNo',
                 'exprun_TaxonID', 'exprun_nTechRepeats']
    for field in intfields:
        ispecdata[field] = ispecdata[field].fillna(0)
        ispecdata[field] = ispecdata[field].astype('int')  # get rid of decimals

    for col in [c for c in ispecdata.columns if c not in intfields]:
        ispecdata[col] = ispecdata[col].fillna('')

    for ix, row in ispecdata.iterrows():
        if isinstance(row.exprun_CreationTS, str):
            try:
                creation = datetime.strptime(row.exprun_CreationTS, '%m/%d/%Y %H:%M:%S')
            except ValueError:
                creation = None
        else:
            creation = row.exprun_CreationTS

        passflag = True
        if norepeats:
            if row.exprun_EXPRecNo in d:
                if row.exprun_EXPRunNo in d.get(row.exprun_EXPRecNo):
                    passflag = False

        if passflag:
            newexps[row.exprun_EXPRecNo].append(
                {'run_no':row.exprun_EXPRunNo,
                 'search_no': row.exprun_EXPSearchNo, 'taxon': row.exprun_TaxonID,
                 'addedby': row.exprun_AddedBy, 'creation_ts': creation,
                 'purpose': row.exprun_Purpose, 'label': row.exprun_LabelType,
                 'techrep': row.exprun_nTechRepeats, 'instrument':row.exprun_MS_Instrument,
                 'msexperimenter': row.exprun_MS_Experimenter,
                 'mscomment':row.exprun_MS_Experimenter,
                 'quant': row.exprun_Search_QuantSource,
                 'searchexperimenter': row.exprun_Search_Experimenter}
                )


    if todb and newexps:
        print('Updating experiment records')
        add_experiments(newexps)
    elif not todb:
        return newexps

if __name__ == '__main__':
    ispecdir = 'C:\\Users\\saltzman\\Desktop\\testing'
    ispecf = '4PyGrouper_ExpRunDump.xlsx'
    f = os.path.join(ispecdir, ispecf)
    get_ispec_experiment_info(f, todb=True, norepeats=True)  # add only new entries to database

