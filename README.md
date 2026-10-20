# Lunar regolith melt viscosity: Earth tests vs Moon conditions

Independent project. How much do sintering- and melt-processing temperatures shift between
Earth tests (simulants melted in air, Fe³⁺/Feₜₒₜ ≈ 0.6) and lunar conditions (IW−1,
Fe³⁺/Feₜₒₜ ≈ 0.02), soil by soil, and where are current models too uncertain to say?

See `RESULTS.md` for findings. Everything below reproduces them.

## Run order

```bash
pip install numpy pandas scipy scikit-learn torch openpyxl pdfplumber pypdfium2 gpvisc==0.3.5
git clone --depth 1 https://github.com/charlesll/gpvisc.git data/raw/gpvisc/repo
# open-access PDFs into data/raw/papers/ (see "Data sources"), Zenodo 5899696 into data/raw/zenodo_5899696/

python src/extract_stapperfend.py     # held-out simulant test set (16 melts)
python src/data.py                    # training table from the gpvisc lp sheet
python src/redox_data.py              # measured controlled-fO2 series (Chevrel 2013/2014)
python src/lunar_soils.py             # application set (66 returned soils)
python src/reproduce_stapperfend.py   # step 2: validate pipeline against published numbers
python src/validate_sw2016.py         # SW2016 implementation vs paper + Stapperfend
python src/run_experiments.py         # step 3: grouped CV, full fit, simulant scorecard
python src/redox_test.py              # step 4: redox stress test
python src/earth_vs_moon.py           # deliverables 2 and 3
python src/figures.py                 # figures
```

## Layout

| File | What it does |
|---|---|
| `src/chem.py` | wt%→mol, FeOT, Borisov 2018 / Kress & Carmichael 1991 redox, IW buffer, NBO/T |
| `src/extract_stapperfend.py` | Pulls compositions, viscosities, VFT fits and published model scores from both 2025 papers |
| `src/data.py` | Anhydrous gpvisc `lp` rows, redox-invariant composition groups, held-out redox studies |
| `src/redox_data.py` | Controlled-fO₂ viscosity and Tg series with titrated Fe²⁺/Feₜₒₜ |
| `src/lunar_soils.py` | Apollo / Luna / Chang'e-5 soil chemistry |
| `src/baselines.py` | GRD 2008 (own implementation), pretrained gpvisc 0.3.5 |
| `src/sw2016.py` | Sehlke & Whittington 2016 + corrigendum, both iron conventions; `src/validate_sw2016.py` checks it |
| `src/models.py` | Ridge (NBO/T features), gradient boosting, VFT-output MLP, deep ensemble |
| `src/zoo.py` | One prediction interface over all models; iso-viscosity temperature solver |

## Data sources

| Role | Source | Access |
|---|---|---|
| Training | gpvisc `database.xlsx`, sheet `lp` (Le Losq et al. 2025) | GitHub charlesll/gpvisc |
| Held-out lunar test | Stapperfend et al. 2025, Am. Mineral. 110:1171 | open access (msaweb.org) |
| Held-out lunar test | Stapperfend et al. 2025, Chem. Geol. 696:123082 | open access (depositonce.tu-berlin.de) |
| Redox test | Chevrel 2013 PhD thesis, LMU (= Chevrel et al. 2013 GCA, 2014 GCA) | open access (edoc.ub.uni-muenchen.de/15783) |
| Application | Zenodo 5899696, Table S2 (Apollo, Luna, Chang'E-5 soils) | open |
| Baseline | Sehlke & Whittington 2016 GCA 191 + corrigendum GCA 197 | library access (PDF not committed: paywalled) |
| Not obtained | Cukierman & Uhlmann 1974 JGR 79 (lunar 15555 redox series) | paywalled |
