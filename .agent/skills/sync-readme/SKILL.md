---

name: sync-readme

description: Đồng bộ README của môn NLP với nội dung lý thuyết và thực hành/lab thực tế có trong repository.

---

# Skill làm practice
- Đối với từng lab, làm bài tập theo yêu cầu của file pdf lab đó.
- Đọc kĩ nội dung yêu cầu cần hoàn thành của các file
- Đối với các phần cần hoàn thành, cần ghi rõ tiêu đề, phần nào, câu nào. Làm đúng theo yêu cầu đề bài, giải thích ... 

# Skill Đồng bộ README

## Mục đích

Giữ bảng **`Syllabus & Progress`** trong `README.md` được đồng bộ với các tài liệu môn học thực tế có trong:

* `theory/`
* `practice/`

## Quy ước cấu trúc Project

Repository tuân theo cấu trúc:

```text
natural_language_processing/

├── README.md
├── requirements.txt
├── .gitignore
├── theory/
│   ├── week1/                        # Tài liệu lý thuyết, slide, corpus của Tuần 1
│   └── weekX/                        # Các tuần tiếp theo
└── practice/
    ├── lab01/                        # Thư mục Lab 01
    └── labXX/                        # Các lab tiếp theo
```

## Quy trình & Quy tắc

### 1. Kiểm tra Repository trước

Trước khi cập nhật `README.md`:

1. Kiểm tra `theory/` để tìm các thư mục `week<N>`, slide bài giảng, dataset hoặc ghi chú.
2. Kiểm tra `practice/` để tìm các thư mục `lab<NN>` (ví dụ: `lab01`, `lab02`).
3. Kiểm tra tiêu đề notebook, ghi chú Markdown (`calculations.md`) và các file PDF (`W<N>.pdf`) để xác định chính xác các từ khóa/chủ đề được học.
4. Kiểm tra nội dung hiện tại và số dòng của section `## 📚 Syllabus & Progress` trong `README.md`.

> [!IMPORTANT]
> **Không được đoán hoặc tự tạo các chủ đề không có file tương ứng hoặc không có cơ sở từ syllabus chính thức. Chỉ đánh dấu những nội dung thực sự tồn tại hoặc được nêu chính thức trong chương trình học.**

### 2. Chỉ cập nhật section Syllabus & Progress

* **KHÔNG được ghi đè hoặc chỉnh sửa các section khác** của `README.md`, chẳng hạn:

  * `Course Information`
  * `Environment Setup`
  * `Useful References & Resources`
  * và các section khác ngoài `Syllabus & Progress`.
* Chỉ cập nhật các hàng trong bảng thuộc section:

```markdown
## 📚 Syllabus & Progress
```

### 3. Schema & Format của bảng

Bảng **PHẢI** tuân theo chính xác schema 4 cột sau:

```markdown
| Tuần | Chủ đề lý thuyết | Thực hành / Lab | Trạng thái |
| :---: | :--- | :--- | :---: |
| **01** | **Giới thiệu về NLP**; Biểu diễn vector thưa & vector dày; Chuẩn hóa Count Vector; TF-IDF; Pipeline xử lý văn bản; Demo trên Corpus 30K tài liệu | [**Lab 01 – Từ xử lý văn bản đến tìm kiếm**](./practice/lab01/) | 🟡 Đang thực hiện |
```

#### Quy định từng cột

* **`Tuần`**:

  * Phải là chuỗi 2 chữ số.
  * Được in đậm.
  * Ví dụ: `**01**`, `**02**`, `**03**`.

* **`Chủ đề lý thuyết`**:

  * Liệt kê các từ khóa/chủ đề lý thuyết cốt lõi.
  * Các chủ đề được phân cách bằng dấu chấm phẩy `;`.
  * Chỉ sử dụng các chủ đề có cơ sở từ tài liệu thực tế hoặc syllabus chính thức.

* **`Thực hành / Lab`**:

  * Phải là Markdown link hợp lệ tới thư mục lab tương ứng.
  * Format:

```markdown
[**Lab <NN> – <Tên Lab>**](./practice/lab<NN>/)
```

* Ví dụ:

```markdown
[**Lab 01 – Từ xử lý văn bản đến tìm kiếm**](./practice/lab01/)
```

* **`Trạng thái`**:

  * Chỉ sử dụng một trong ba trạng thái sau:

  * `🟢 Hoàn thành`: Tất cả bài tập và phần tính toán bắt buộc trong lab đã hoàn thành và được kiểm tra.

  * `🟡 Đang thực hiện`: Nội dung hiện đang được học hoặc đang trong quá trình thực hiện.

  * `⏳ Sắp tới`: Nội dung được lên kế hoạch cho các tuần tiếp theo.

### 4. Kiểm tra sau khi cập nhật

Sau khi cập nhật `README.md`:

1. Kiểm tra tất cả relative link trong bảng có trỏ tới thư mục thực sự tồn tại hay không.

   * Ví dụ: `./practice/lab01/`
   * `./practice/lab02/`
2. Đảm bảo bảng Markdown có cú pháp hợp lệ.
3. Đảm bảo không có delimiter `|` bị thiếu hoặc thừa.
4. Đảm bảo bảng hiển thị chính xác khi render Markdown.
5. Đảm bảo **chỉ section `Syllabus & Progress` được thay đổi**, các section khác của `README.md` phải được giữ nguyên.
6. Đối chiếu lại nội dung với file thực tế trong `theory/` và `practice/` trước khi xác nhận hoàn thành.


