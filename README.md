# OCR Document Extraction: End-to-End Pipeline with Faster R-CNN & CNN-Transformer

## Giới thiệu (Overview)

Dự án này xây dựng một hệ thống OCR (Optical Character Recognition) hai giai đoạn (Two-stage) để trích xuất văn bản từ ảnh tài liệu (Form Understanding), được tối ưu hóa trên bộ dữ liệu **FUNSD**. Hệ thống kết hợp các kỹ thuật Computer Vision hiện đại để phát hiện vùng chữ và mô hình Sequence-to-Sequence để nhận dạng văn bản.

-----

##  Kiến trúc hệ thống (System Architecture)

Hệ thống hoạt động theo quy trình Pipeline 2 bước: **Text Detection** $\rightarrow$ **Text Recognition**.

### 1\. Text Detection Module (Phát hiện vùng chữ)

Mục tiêu: Xác định vị trí (Bounding Box) của các vùng chứa văn bản trong tài liệu.

  * **Model:** Faster R-CNN.
  * **Backbone:** ResNet50 + FPN (Feature Pyramid Network).
      * *Lý do kỹ thuật:* FPN giúp mô hình trích xuất đặc trưng ở nhiều tỷ lệ khác nhau, cải thiện đáng kể khả năng phát hiện các dòng văn bản nhỏ hoặc văn bản nằm rải rác trong biểu mẫu.
  * **Prediction Head:** FastRCNNPredictor được tinh chỉnh cho 2 lớp (Background và Text).
  * **Loss Function:** Kết hợp giữa RPN Loss (Objectness + Box regression) và R-CNN Loss (Classifier + Box regression).

### 2\. Text Recognition Module (Nhận dạng văn bản)

Mục tiêu: Chuyển đổi vùng ảnh đã crop thành chuỗi ký tự (Sequence generation).

Kiến trúc lai **CNN + Transformer Decoder**:

  * **Encoder (Visual Feature Extraction):**
      * Sử dụng **ResNet18** (đã loại bỏ lớp FC và Pooling cuối).
      * Trích xuất đặc trưng không gian (Spatial Features) từ ảnh đầu vào.
      * **Patch Embedding:** Chiếu đặc trưng từ CNN $(B, C, H, W)$ thành chuỗi vector $(B, S, E)$ để làm đầu vào cho Transformer.
  * **Decoder (Sequence Modeling):**
      * Kiến trúc: **Transformer Decoder** thuần túy.
      * Cơ chế: **Masked Self-Attention** (đảm bảo tính tuần tự) và **Cross-Attention** (liên kết giữa đặc trưng ảnh từ Encoder và từ đang sinh).
      * **Positional Encoding:** Sử dụng Learnable Embedding để mô hình hóa thứ tự chuỗi.
  * **Decoding Strategy:** Greedy Decoding (trong quá trình Inference).

-----

##  Dataset & Preprocessing

Dự án sử dụng bộ dữ liệu **FUNSD** (Form Understanding in Noisy Scanned Documents) thông qua thư viện `python-doctr`.

  * **Detection Data:**
      * Resize ảnh về kích thước cố định `(640, 640)`.
      * Chuẩn hóa Bounding Box coordinates.
  * **Recognition Data:**
      * Crop từng vùng văn bản dựa trên Ground Truth (Train) hoặc Predicted Box (Inference).
      * Resize về `(224, 224)` và chuẩn hóa theo ImageNet stats (`mean=[0.485, ...], std=[0.229, ...]`).
      * Tokenization: Xây dựng bộ từ điển ký tự (Char-level vocab) bao gồm các token đặc biệt `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`.

-----
