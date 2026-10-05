# Quy tắc Workspace: MAT3561 - Xử lý ngôn ngữ tự nhiên và ứng dụng

Tài liệu này định nghĩa các quy tắc repository, quy ước và nguyên tắc vận hành dành cho AI assistant khi làm việc trong workspace này.

---

## 1. Bối cảnh Project & Môn học

* **Môn học**: MAT3561 - Natural Language Processing and Applications (Xử lý ngôn ngữ tự nhiên và ứng dụng)

* **Giảng viên lý thuyết**: Dr. Le Hong Phuong (Phòng 105-T5) — [www.phuong.pro](http://www.phuong.pro)

* **Giảng viên thực hành**: M.Sc. Pham Ngoc Hai (Phòng máy PM)

* **Thông tin sinh viên**:

  * Họ và tên: **Vũ Tiến Đạt**
  * Mã sinh viên (MSSV): `23000111`

---

## 2. Quy ước cấu trúc thư mục

```text
natural_language_processing/

├── README.md                          # Dashboard chính và bảng theo dõi syllabus
├── requirements.txt                   # Các dependency của project
├── .gitignore                         # Quy tắc loại trừ của Git
├── AGENTS.md                          # Quy tắc và hướng dẫn dành cho AI assistant
├── .agent/skills/sync-readme/SKILL.md # Skill đồng bộ syllabus với các file thực tế
├── theory/
└── practice/

```

---

## 3. Quy ước đặt tên & Coding Standards

### Quy ước đặt tên file

* **Thư mục lý thuyết**: `theory/week<N>`

  * Ví dụ: `theory/week1`, `theory/week2`

* **Thư mục thực hành**: `practice/lab<NN>`

  * Ví dụ: `practice/lab01`, `practice/lab02`

### Quy ước Python & Machine Learning

* **Phiên bản Python**: `>= 3.10`

* **Dependencies**: Sử dụng các package được khai báo trong `requirements.txt`, bao gồm:

  * PyTorch
  * Transformers
  * NLTK
  * Spacy
  * Underthesea
  * PyVi
  * Scikit-learn

* **Khả năng tái lập (Reproducibility)**:

  * Luôn cố định random seed để đảm bảo kết quả có thể tái lập:

    ```python
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    ```

* **Chất lượng Notebook**:

  * Sử dụng
