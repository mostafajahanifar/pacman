import os

### ========== PATHS ==========
# Get the repo root (two levels up from config.py)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


for d in [DATA_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

### ========== HYPERPARAMETERS ==========

### ========== DOMAIN CONSTANTS ==========
ALL_CANCERS = sorted(['SARC',
    'LIHC',
    'THYM',
    'ACC',
    'BRCA',
    'KICH',
    'STAD',
    'BLCA',
    'THCA',
    'GBMLGG',
    'UCEC',
    'LUAD',
    'KIRC',
    'KIRP',
    'PAAD',
    'CESC',
    'PCPG',
    'MESO',
    'SKCM',
    'PRAD',
    'COADREAD',
    'ESCA',
    'LUSC',
    'HNSC',
    'OV',
    'TGCT',
    'CHOL',
    'DLBC',
    'UCS'
 ])

SURV_CANCERS = sorted(["ACC", "BLCA", "BRCA", "CESC", "COADREAD", "ESCA", "GBMLGG", "HNSC", "KIRC", "KIRP", "LIHC", "LUAD", "LUSC", "OV", "PAAD", "SKCM", "STAD", "UCEC", "MESO", "PRAD", "SARC", "TGCT", "THCA", "KICH"])

ETHNICITIES_DICT = {
    'WHITE': 'White',
    'BLACK OR AFRICAN AMERICAN': 'Black',
    'ASIAN': 'Asian',
    'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER': 'Pacific\nIslander',
    'AMERICAN INDIAN OR ALASKA NATIVE': 'Native\nAmerican'
}