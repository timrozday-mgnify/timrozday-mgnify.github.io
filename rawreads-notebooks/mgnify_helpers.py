"""Helpers shared by the RQ notebooks. All MGnify access goes through mgnipy.

    pip install mgnipy==0.3.0 scipy matplotlib
"""

import json
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from mgnipy import MGnipy

CACHE = Path(__file__).resolve().parent.parent / "data/notebook_cache"  # downloads land here, so a re-run is fast
# mgnipy caches API responses here too. Shipped with a browser build, the notebooks' own queries need no
# API calls, so they run from any website (the MGnify API only allows a few origins). Delete to refresh.
MG = MGnipy(cache_dir=CACHE / "mgnipy")
RAW = {"Metagenomic", "Metatranscriptomic"}
pd.set_option("display.max_colwidth", 80)


def _pmap(f, xs):
    """Map over xs in 8 threads; one at a time in the browser (Pyodide has no threads)."""
    if sys.platform == "emscripten":
        return list(map(f, xs))
    with ThreadPoolExecutor(8) as pool:
        return list(pool.map(f, xs))


def related(study, resource):
    """A study's analyses or samples as a flat DataFrame."""
    st = MG.study(study)
    st.get()
    lst = st.get_list(resource, fetch=False)
    lst.get_all(hide_progress=True)
    return lst.search_results.to_pandas(expand_nested_dicts=True)


def rawread_analyses(study):
    """A study's V6 raw-read analyses, with their run and sample accessions."""
    an = related(study, "analyses")
    an = an[an.pipeline_version.str.startswith("V6") & an.experiment_type.isin(RAW)]
    an = an.rename(columns={"accession": "analysis", "run__accession": "run", "sample__accession": "sample"})
    return an.reindex(columns=["analysis", "run", "sample", "experiment_type"]).reset_index(drop=True)


def v6_rawread_studies():
    """Studies with V6 raw-read analyses (the pipeline filter also matches amplicon/assembly)."""
    f = CACHE / "v6_rawread_studies.csv"
    if not f.exists():
        studies = MG.studies(has_analyses_from_pipeline="V6")
        studies.get_all(hide_progress=True)
        df = studies.search_results.to_pandas(expand_nested_dicts=True)
        df = df[["accession", "title", "biome__lineage"]].rename(columns={"biome__lineage": "biome"})
        n = [len(rawread_analyses(acc)) for acc in df.accession]
        f.parent.mkdir(parents=True, exist_ok=True)
        df.assign(rawread_analyses=n).query("rawread_analyses > 0").to_csv(f, index=False)
    return pd.read_csv(f)


def samples(study):
    """MGnify sample metadata (ENA's standard fields only), one row per sample."""
    sm = related(study, "samples")
    sm.columns = sm.columns.str.removeprefix("metadata__")
    return sm.set_index("accession")


def _analysis(acc):
    a = MG.analysis(acc)
    a.get()
    return a


def qc(acc):
    """Read counts from the analysis' quality_control_summary.

    fastp: raw reads -> QC-passed reads. The decontam report starts from the reads
    left after host/phiX removal, so its before_filtering count is the clean reads.
    """
    f = CACHE / "qc" / f"{acc}.json"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        rec = next(iter(_analysis(acc).results.values()))[0]
        f.write_text(json.dumps(rec["quality_control_summary"]))
    q = json.loads(f.read_text())
    passed = q["fastp"]["after_filtering"]["total_reads"]
    clean = q["decontam"]["before_filtering"]["total_reads"]
    return {"analysis": acc, "raw_reads": q["fastp"]["before_filtering"]["total_reads"],
            "qc_reads": passed, "clean_reads": clean, "removed_decontam": 1 - clean / passed}


def qc_table(analyses):
    return analyses.merge(pd.DataFrame(_pmap(qc, analyses.analysis)), on="analysis")


def _read_tsv(f, **kw):
    gz = f.read_bytes()[:2] == b"\x1f\x8b"  # per-run tables are gzipped despite the .tsv name
    return pd.read_csv(f, sep="\t", compression="gzip" if gz else None, **kw)


def run_table(acc, profile):
    """One analysis' taxonomy table (motus, silva-ssu or silva-lsu) as counts per lineage."""
    f = CACHE / profile / f"{acc}.tsv"
    if not f.exists():
        _analysis(acc).datasets.download(f.parent, alias=profile, filename=f.name, hide_progress=True)
    t = _read_tsv(f)
    lineage = t.iloc[:, 1:].apply(lambda r: ";".join(r.dropna()), axis=1)
    return t["# Count"].groupby(lineage).sum()


def study_summary(study, profile):
    """The study's V6 summary (lineage x run), or None if MGnify has none.

    Some studies list a summary that is missing on the FTP (404): also None.
    """
    f = CACHE / "summaries" / f"{study}_{profile}.tsv"
    if not f.exists():
        st = MG.study(study)
        st.get()
        dl = st.datasets.downloads_df()
        alias = dl.alias[dl.alias.str.endswith(f"_{profile}_study_summary.tsv")]
        if alias.empty:
            return None
        try:
            st.datasets.download(f.parent, alias=alias.iloc[0], filename=f.name, hide_progress=True)
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 404:
                raise
            return None
    return _read_tsv(f, index_col="taxonomy")


def profiles(study, profile, analyses):
    """Counts, lineage x run: the study summary if there is one, else per-run tables."""
    m = study_summary(study, profile)
    if m is None:
        cols = _pmap(lambda a: run_table(a, profile), analyses.analysis)
        m = pd.concat(cols, axis=1, keys=analyses.run).fillna(0)
    return m[[r for r in analyses.run if r in m.columns]]


def by_rank(m, rank="g"):
    """Sum counts to one rank (g, f, p...). Lineages that stop above it are dropped."""
    name = pd.Series(m.index).str.extract(rf"(?:^|;){rank}__([^;]+)", expand=False)
    return m.groupby(name.values).sum()


def ena_runs(project, fields="run_accession,sample_accession,sample_alias,read_count,base_count"):
    url = f"https://www.ebi.ac.uk/ena/portal/api/filereport?accession={project}&result=read_run&fields={fields}"
    return pd.read_csv(url, sep="\t")


def ena_attributes(accs):
    """Every submitter attribute from the ENA sample XML, one row per sample."""
    accs, rows = list(accs), {}
    for i in range(0, len(accs), 100):
        url = "https://www.ebi.ac.uk/ena/browser/api/xml/" + ",".join(accs[i:i + 100])
        for s in ET.parse(urllib.request.urlopen(url)).iter("SAMPLE"):
            attrs = {a.findtext("TAG"): a.findtext("VALUE") for a in s.iter("SAMPLE_ATTRIBUTE")}
            for acc in {s.get("accession")} | {e.text for e in s.iter("EXTERNAL_ID")}:
                rows[acc] = {"alias": s.get("alias"), **attrs}
    return pd.DataFrame.from_dict(rows, orient="index").reindex(accs)


def pcoa(rel):
    """Bray-Curtis PCoA of a runs x taxa table. Returns 2 axes and their % variance."""
    d = squareform(pdist(rel.values, "braycurtis"))
    n = len(d)
    j = np.eye(n) - 1 / n
    w, v = np.linalg.eigh(-0.5 * j @ (d ** 2) @ j)
    w, v = w[::-1], v[:, ::-1]
    xy = pd.DataFrame(v[:, :2] * np.sqrt(w[:2]), index=rel.index, columns=["PC1", "PC2"])
    return xy, 100 * w[:2] / w[w > 0].sum()


def shannon(m):
    p = m / m.sum()
    return -(p * np.log(p.where(p > 0))).sum()
