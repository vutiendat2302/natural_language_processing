---
name: sync-readme
description: Synchronize the NLP course README with theory and practical/lab content in the repository.
---

# Sync README Skill

## Purpose

Keep the `Syllabus & Progress` table in `README.md` synchronized with the actual course materials present in:
- `theory/`
- `practice/`

## Project Structure Conventions

The repository follows this layout:

```text
natural_language_processing/
├── README.md
├── requirements.txt
├── .gitignore
├── theory/
│   ├── ed3book_aug26.pdf             # Reference textbook (Jurafsky & Martin)
│   ├── hf_transformers_tutorial.pdf  # Tutorial references
│   ├── week1/                        # Theory materials, slides, corpora for Week 1
│   └── weekX/                        # Subsequent weeks
└── practice/
    ├── lab01/                        # Lab 01 directory
    │   ├── W1.pdf                    # Lab instructions/prompts
    │   ├── calculations.md           # Theoretical & numerical solutions
    │   └── <MSSV>_<Name>_BT1.ipynb   # Student implementation notebook
    └── labXX/                        # Subsequent labs
```

## Workflow & Rules

### 1. Inspect the Repository First

Before updating `README.md`:
1. Check `theory/` for `week<N>` folders, lecture slides, datasets, or notes.
2. Check `practice/` for `lab<NN>` folders (e.g. `lab01`, `lab02`).
3. Examine notebook titles, markdown notes (`calculations.md`), and PDFs (`W<N>.pdf`) to extract precise topic keywords.
4. Verify the existing content and line numbers of the `Syllabus & Progress` section in `README.md`.

> [!IMPORTANT]
> Never guess or invent topics that have no corresponding files or syllabus basis. Only mark items that exist or are officially outlined.

### 2. Update Only the Syllabus & Progress Section

- **Do NOT overwrite or modify other sections** of `README.md` (such as `Course Information`, `Environment Setup`, or `Useful References & Resources`).
- Update the table rows under `## 📚 Syllabus & Progress`.

### 3. Table Schema & Formatting

The table MUST follow this exact 4-column schema:

```markdown
| Week | Theory Topics | Practical / Lab | Status |
| :---: | :--- | :--- | :---: |
| **01** | **Introduction to NLP**; Sparse & Dense Vector Representations; Count Vector Normalization; TF-IDF; Document Processing Pipeline; Demo on the 30K-Document Corpus | [**Lab 01 – From Text Processing to Search**](./practice/lab01/) | 🟡 In Progress |
```

- **Week**: 2-digit bold string (`**01**`, `**02**`, etc.).
- **Theory Topics**: Core theory keywords separated by semicolons.
- **Practical / Lab**: Clean Markdown link pointing to the lab directory (`[**Lab <NN> – <Title>**](./practice/lab<NN>/)`).
- **Status indicators**:
  - `🟢 Completed`: All required exercises and calculations in the lab are done and verified.
  - `🟡 In Progress`: Currently being worked on.
  - `⏳ Upcoming`: Planned for future weeks.

### 4. Verification

After updating `README.md`:
- Verify all relative links point to existing directories (`./practice/lab01/`, etc.).
- Ensure Markdown table syntax renders cleanly without broken delimiters.