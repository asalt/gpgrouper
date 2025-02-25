# mapper.py
import sqlitedict
import json
from pathlib import Path
from mygene import MyGeneInfo
import re
import logging

logger = logging.getLogger(__name__)


# Set data directory
def get_data_dir():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    if not data_dir.exists():
        data_dir.mkdir()
    return data_dir


data_dir = get_data_dir()
sqlitedict_filename = data_dir / "mappings.sqlite"


def get_db(table_name="id_to_uniprot"):
    """
    Open and return an SQLiteDict connection for a given table.
    Different tables can store different mappings.
    """
    return sqlitedict.SqliteDict(
        str(sqlitedict_filename),
        tablename=table_name,
        autocommit=False,  # Manual commit for performance
        encode=json.dumps,
        decode=json.loads,
    )


def update_db(table_name, items):
    """
    Update SQLiteDict with newly fetched gene mappings.
    Stores UniProt → Gene info.
    """
    db = get_db(table_name)
    for item in items:
        if "query" in item:
            db[item["query"]] = item
    db.commit()


def fetch_info_from_db(keys, key_type="ensembl.protein", fetch_online=True):
    """
    Fetch information from a specified SQLiteDict table.
    """
    db = get_db(key_type)
    results = {k: db.get(k, None) for k in keys if k in db}
    missing_keys = set(keys) - set(results.keys())
    logger.debug(f"Found {len(results)} mappings in the database.")
    logger.debug(f"Missing {len(missing_keys)} mappings.")

    if missing_keys and fetch_online:
        new_info = fetch_info_online(missing_keys, key_type)
        update_db(key_type, new_info)
        results.update(
            {item["query"]: item for item in new_info}
        )  # this might not be right

    db.close()
    return results


def fetch_info_online(missing_ids, key_type="ensembl.protein"):
    """
    Query MyGene API for missing identifier mappings.
    key_type determines what ID type to search (default: ENSP).
    """
    if not missing_ids:
        return []

    mg = MyGeneInfo()
    results = mg.querymany(
        missing_ids,
        scopes=key_type,  # e.g., "ensembl.protein", "ensembl.gene", "entrezgene"
        fields="uniprot.Swiss-Prot,uniprot.TrEMBL,name,symbol,other_names,entrezgene,taxid",
        species="all",
        returnall=True,
    )
    out_results = results.get("out", [])

    if results.get("dup"):
        logging.info("handling duplicates")
        for dup_key, dup_count in results["dup"]:
            matches = [x for x in out_results if x["query"] == dup_key]
            # min_geneid = min(matches, key=lambda x: len(x.get("entrezgene", {}), default={}))
            min_match_geneid = min(
                matches, key=lambda x: len(x.get("entrezgene", x.get("query", "")))
            )
            out_results = [x for x in out_results if x["query"] != dup_key]
            out_results.append(min_match_geneid)

    return out_results
