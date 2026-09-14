# Workspace Rules: MAT3561 - Natural Language Processing & Applications

This document defines repository guidelines, conventions, and operational rules for AI assistants working in this workspace.

---

## 1. Project & Course Context
- **Course**: MAT3561 - Natural Language Processing and Applications (Xử lý ngôn ngữ tự nhiên và ứng dụng)
- **Lecturer (Theory)**: Dr. Le Hong Phuong (Room 105-T5) — [www.phuong.pro](http://www.phuong.pro)
- **Instructor (Practice)**: M.Sc. Pham Ngoc Hai (Computer Lab PM)
- **Student Profile**:
  - Full Name: **Vũ Tiến Đạt**
  - Student ID (MSSV): `23000111`

---

## 2. Directory Structure Conventions

```text
natural_language_processing/
├── README.md                          # Main course dashboard & syllabus tracker
├── requirements.txt                   # Project dependencies
├── .gitignore                         # Git exclusion rules
├── AGENTS.md                          # Persistent agent rules and guidelines
├── .agent/skills/sync-readme/SKILL.md # Skill to sync syllabus with actual files
├── theory/
│   ├── ed3book_aug26.pdf              # Core textbook (Jurafsky & Martin)
│   ├── hf_transformers_tutorial.pdf   # Transformers reference
│   └── week<N>/                       # Theory slides, lecture notes, corpora
└── practice/
    └── lab<NN>/                       # e.g., lab01, lab02, ...
        ├── W<N>.pdf                   # Lab assignment problem statement
        ├── <MSSV>_<FullName>_BT<N>.ipynb  # Implementation notebook
        └── calculations.md            # Detailed math solutions & derivations
```

---

## 3. Naming & Coding Standards

### File Naming Conventions
- **Practice Notebooks**: `<MSSV>_<FullNameWithoutAccent>_BT<N>.ipynb`
  - Example: `23000111_VuTienDat_BT1.ipynb`
- **Theory Directories**: `theory/week<N>` (e.g. `theory/week1`, `theory/week2`)
- **Practice Directories**: `practice/lab<NN>` (e.g. `practice/lab01`, `practice/lab02`)

### Python & Machine Learning Conventions
- **Python Version**: `>= 3.10`
- **Dependencies**: Rely on packages specified in `requirements.txt` (PyTorch, Transformers, NLTK, Spacy, Underthesea, PyVi, Scikit-learn).
- **Reproducibility**: Always fix random seeds for deterministic results (`random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`).
- **Notebook Quality**:
  - Use clear Markdown cells to explain formulas, intuition, and algorithm steps using LaTeX.
  - Do not leave empty error tracebacks or unexecuted required cells.
  - Format plots with proper titles, labels, and legends.

---

## 4. Language & Communication Policy

- **Chat & Discussions**: Respond to the user in **Vietnamese** (natural, concise, technical).
- **Code, Comments & Documentation**: Write all code comments, docstrings, Markdown reports, and `README.md` content in **English** unless explicitly requested otherwise.

---

## 5. Git & Data Management Rules

- **Never commit large binaries**: Model checkpoints (`*.pt`, `*.pth`, `*.bin`, `*.safetensors`), large datasets (`*.jsonl`, `*.parquet`), or embedding dumps (`glove*`, `word2vec*`).
- Always respect [`.gitignore`](file:///home/datbritget/Documents/natural_language_processing/.gitignore).
- When modifying `README.md`, only update the `Syllabus & Progress` table or specifically requested sections; do not wipe custom user content.
