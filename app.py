"""
⚡ RedoxScreen — Redox Flow Battery Molecule Screener
AI-powered cheminformatics tool to identify candidate molecules
for next-generation redox flow batteries.

Author: Aryan Metwate
Stack : RDKit · scikit-learn · Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen
from rdkit.Chem.Draw import rdMolDraw2D

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RedoxScreen ⚡",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1 { color: #FFD700; font-size: 2.8rem; margin: 0; }
    .main-header p  { color: #aaa; font-size: 1rem; margin: 0.5rem 0 0; }

    .result-good {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        padding: 1.5rem; border-radius: 12px; text-align: center;
        color: white; font-size: 1.4rem; font-weight: bold;
        box-shadow: 0 4px 20px rgba(56,239,125,0.3);
    }
    .result-bad {
        background: linear-gradient(135deg, #c0392b, #e74c3c);
        padding: 1.5rem; border-radius: 12px; text-align: center;
        color: white; font-size: 1.4rem; font-weight: bold;
        box-shadow: 0 4px 20px rgba(231,76,60,0.3);
    }
    .stSidebar { background: #0d0d1a; }
    .info-box {
        background: #1a1a2e; border-left: 4px solid #FFD700;
        padding: 1rem; border-radius: 8px; margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state init ─────────────────────────────────────────────────────────
if "smiles" not in st.session_state:
    st.session_state.smiles = "O=C1c2ccccc2C(=O)c2ccccc21"  # Anthraquinone default

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("redox_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("features.pkl", "rb") as f:
        features = pickle.load(f)
    return model, features

model, FEATURES = load_model()

# ── Helper functions ───────────────────────────────────────────────────────────
def get_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    desc = {
        "MolWt":               Descriptors.MolWt(mol),
        "LogP":                Crippen.MolLogP(mol),
        "NumHDonors":          rdMolDescriptors.CalcNumHBD(mol),
        "NumHAcceptors":       rdMolDescriptors.CalcNumHBA(mol),
        "NumRings":            rdMolDescriptors.CalcNumRings(mol),
        "NumAromaticRings":    rdMolDescriptors.CalcNumAromaticRings(mol),
        "TPSA":                Descriptors.TPSA(mol),
        "NumRotBonds":         rdMolDescriptors.CalcNumRotatableBonds(mol),
        "FractionCSP3":        rdMolDescriptors.CalcFractionCSP3(mol),
        "NumHeteroatoms":      rdMolDescriptors.CalcNumHeteroatoms(mol),
        "NumValenceElectrons": Descriptors.NumValenceElectrons(mol),
        "MaxPartialCharge":    Descriptors.MaxPartialCharge(mol),
        "MinPartialCharge":    Descriptors.MinPartialCharge(mol),
        "NumRadicalElectrons": Descriptors.NumRadicalElectrons(mol),
        "RingCount":           Descriptors.RingCount(mol),
        "HeavyAtomCount":      mol.GetNumHeavyAtoms(),
        "NumAromaticBonds":    sum(1 for b in mol.GetBonds() if b.GetIsAromatic()),
        "Chi0":                Descriptors.Chi0(mol),
        "Chi1":                Descriptors.Chi1(mol),
        "Kappa1":              Descriptors.Kappa1(mol),
    }
    return desc, mol

def mol_to_svg(mol, size=(350, 280)):
    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    drawer.drawOptions().addStereoAnnotation = True
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()

def predict(smiles):
    desc, mol = get_descriptors(smiles)
    if desc is None:
        return None, None, None, None
    X = pd.DataFrame([desc])[FEATURES].fillna(0)
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    return pred, prob, desc, mol

def redox_score(desc, prob):
    score = prob * 50
    score += min(desc["NumAromaticRings"] * 5, 20)
    score += min(desc["NumHeteroatoms"] * 3, 15)
    if desc["LogP"] < 1.5:
        score += 10
    if desc["MolWt"] < 50:
        score -= 10
    return min(max(round(score), 0), 100)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ RedoxScreen")
    st.markdown("---")
    st.markdown("### 🔬 Example Molecules")
    st.markdown("Click any to auto-fill:")

    examples = {
        "Anthraquinone ✅":  "O=C1c2ccccc2C(=O)c2ccccc21",
        "AQDS ✅":           "O=C1c2cc(S(=O)(=O)O)ccc2C(=O)c2ccc(S(=O)(=O)O)cc21",
        "TEMPO ✅":          "CC1(C)CC(=O)CC(C)(C)N1[O]",
        "Benzoquinone ✅":   "O=C1C=CC(=O)C=C1",
        "Methyl Viologen ✅":"C[n+]1ccc(C=Cc2cc[n+](C)cc2)cc1",
        "Ethanol ❌":        "CCO",
        "Benzene ❌":        "c1ccccc1",
        "Hexane ❌":         "CCCCCC",
    }
    for name, smi in examples.items():
        if st.button(name, key=f"btn_{name}", use_container_width=True):
            st.session_state.smiles = smi

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
    <div class='info-box'>
    Redox Flow Batteries store renewable energy 
    in liquid electrolytes. This app uses ML + RDKit 
    to screen organic molecules for their suitability 
    as electrolyte candidates.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("**Built by:** Aryan Metwate")
    st.markdown("**Stack:** RDKit · sklearn · Streamlit")

# ── Main header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1>⚡ RedoxScreen</h1>
    <p>AI-powered Redox Flow Battery Molecule Screener — Cheminformatics × Clean Energy</p>
</div>
""", unsafe_allow_html=True)

# ── SMILES Input ───────────────────────────────────────────────────────────────
col_input, col_btn = st.columns([4, 1])
with col_input:
    smiles_input = st.text_input(
        "🔬 Enter SMILES string:",
        value=st.session_state.smiles,
        placeholder="e.g. O=C1C=CC(=O)C=C1",
        key="smiles_input_box"
    )
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("🚀 Screen", use_container_width=True, type="primary")

# ── Results ────────────────────────────────────────────────────────────────────
if run or smiles_input:
    smi = smiles_input.strip()
    pred, prob, desc, mol = predict(smi)

    if pred is None:
        st.error("❌ Invalid SMILES! Please check your input and try again.")
    else:
        score = redox_score(desc, prob)

        st.markdown("<br>", unsafe_allow_html=True)
        if pred == 1:
            st.markdown(f"""
            <div class='result-good'>
                ✅ GOOD REDOX CANDIDATE &nbsp;|&nbsp; Confidence: {prob*100:.1f}% &nbsp;|&nbsp; Score: {score}/100
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-bad'>
                ❌ POOR REDOX CANDIDATE &nbsp;|&nbsp; Confidence: {(1-prob)*100:.1f}% not suitable &nbsp;|&nbsp; Score: {score}/100
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Structure + Properties ─────────────────────────────────────────
        left, right = st.columns([1, 1])

        with left:
            st.markdown("### 🧪 2D Molecular Structure")
            svg = mol_to_svg(mol)
            b64 = base64.b64encode(svg.encode()).decode()
            st.markdown(
                f'<img src="data:image/svg+xml;base64,{b64}" width="100%">',
                unsafe_allow_html=True
            )

        with right:
            st.markdown("### 📊 Molecular Properties")
            props = {
                "⚖️ Molecular Weight":    f"{desc['MolWt']:.2f} g/mol",
                "💧 LogP (lipophilicity)": f"{desc['LogP']:.3f}",
                "🔵 H-Bond Donors":        int(desc["NumHDonors"]),
                "🔴 H-Bond Acceptors":     int(desc["NumHAcceptors"]),
                "💍 Aromatic Rings":       int(desc["NumAromaticRings"]),
                "⚛️ Heteroatoms":          int(desc["NumHeteroatoms"]),
                "📐 TPSA":                 f"{desc['TPSA']:.2f} Å²",
                "🔄 Rotatable Bonds":      int(desc["NumRotBonds"]),
                "🔬 Heavy Atoms":          int(desc["HeavyAtomCount"]),
                "⚡ Radical Electrons":    int(desc["NumRadicalElectrons"]),
            }
            for k, v in props.items():
                c1, c2 = st.columns([2, 1])
                c1.markdown(f"**{k}**")
                c2.markdown(f"`{v}`")

        st.markdown("---")

        # ── Score gauge + Battery criteria ────────────────────────────────
        col3, col4 = st.columns([1, 1])

        with col3:
            st.markdown("### 🎯 Redox Suitability Score")
            fig, ax = plt.subplots(figsize=(5, 3))
            fig.patch.set_facecolor("#0d0d1a")
            ax.set_facecolor("#0d0d1a")
            color = "#38ef7d" if score >= 60 else "#e74c3c" if score < 40 else "#f39c12"
            ax.barh(["Score"], [100], color="#1e1e2e", height=0.4, edgecolor="none")
            ax.barh(["Score"], [score], color=color, height=0.4, edgecolor="none")
            ax.set_xlim(0, 100)
            ax.set_xlabel("Score / 100", color="white")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.text(score + 1, 0, f"{score}/100",
                    va="center", ha="left", color="white", fontsize=14, fontweight="bold")
            ax.set_title("Redox Suitability", color="#FFD700", fontsize=13)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col4:
            st.markdown("### 🔋 Battery Suitability Checklist")
            criteria = {
                "Aromatic/Conjugated system":   desc["NumAromaticRings"] >= 2,
                "Has Heteroatoms (O/N)":        desc["NumHeteroatoms"] >= 2,
                "Good Solubility (LogP < 2)":   desc["LogP"] < 2.0,
                "Reasonable MW (< 500 g/mol)":  desc["MolWt"] < 500,
                "Redox Active (ML prediction)": pred == 1,
            }
            for criterion, passed in criteria.items():
                icon = "✅" if passed else "❌"
                st.markdown(f"{icon} {criterion}")

        # ── Radar chart ───────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📡 Property Radar")

        categories = ["MW\n(norm)", "LogP\n(norm)", "Arom.\nRings",
                       "Hetero\natoms", "HBond\nAcceptors", "Redox\nProb"]
        values = [
            min(desc["MolWt"] / 500, 1),
            min(max((desc["LogP"] + 2) / 6, 0), 1),
            min(desc["NumAromaticRings"] / 5, 1),
            min(desc["NumHeteroatoms"] / 8, 1),
            min(desc["NumHAcceptors"] / 8, 1),
            prob
        ]
        values += values[:1]
        N = len(categories)
        angles = [n / float(N) * 2 * 3.14159 for n in range(N)]
        angles += angles[:1]

        fig2, ax2 = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
        fig2.patch.set_facecolor("#0d0d1a")
        ax2.set_facecolor("#0d0d1a")
        ax2.plot(angles, values, "o-", linewidth=2, color="#FFD700")
        ax2.fill(angles, values, alpha=0.25, color="#FFD700")
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(categories, color="white", fontsize=9)
        ax2.set_ylim(0, 1)
        ax2.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax2.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], color="#888", fontsize=7)
        ax2.tick_params(colors="white")
        ax2.spines["polar"].set_color("#444")
        ax2.grid(color="#333", linestyle="--", alpha=0.5)
        ax2.set_title("Molecular Property Profile", color="#FFD700", fontsize=12, pad=20)
        st.pyplot(fig2)
        plt.close()

        # ── Interpretation ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 💡 Interpretation")
        interp = []
        if desc["NumAromaticRings"] >= 2:
            interp.append("✅ **Multiple aromatic rings** — good π-electron conjugation for stable redox cycling")
        if desc["NumHeteroatoms"] >= 2:
            interp.append("✅ **Heteroatoms present** — oxygen/nitrogen groups enable electrochemical activity")
        if desc["LogP"] < 2:
            interp.append("✅ **Low LogP** — relatively water-soluble, suitable for aqueous flow batteries")
        if desc["NumRadicalElectrons"] > 0:
            interp.append("✅ **Radical electrons detected** — strong indicator of redox activity (like TEMPO)")
        if desc["MolWt"] > 500:
            interp.append("⚠️ **High molecular weight** — may reduce solubility in battery electrolyte")
        if desc["LogP"] > 3:
            interp.append("⚠️ **High LogP** — lipophilic molecule, may have poor water solubility")
        if not interp:
            interp.append("ℹ️ No strong structural indicators found for redox activity")
        for line in interp:
            st.markdown(line)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<center>
<small>
⚡ <b>RedoxScreen</b> — Built with RDKit · scikit-learn · Streamlit &nbsp;|&nbsp;
🌍 Helping accelerate clean energy storage research &nbsp;|&nbsp;
👨‍💻 Aryan Metwate, M.Sc. Bioinformatics
</small>
</center>
""", unsafe_allow_html=True)
