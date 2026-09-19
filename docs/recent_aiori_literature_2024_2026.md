# Ghi chép literature AiOIR và composite restoration (2024–2026)

Cập nhật: 2026-09-20. Tài liệu này lưu lại những gì đã kiểm tra trong quá
trình chọn hướng nghiên cứu, để các phiên sau không phải tìm lại từ đầu.

## Quy ước sử dụng

- Chỉ coi một claim đã được xác nhận khi đã đọc nguồn chính: paper HTML/PDF,
  proceedings chính thức hoặc repository của tác giả.
- “Không thấy” trong danh sách này không có nghĩa là “chưa ai làm”. Trước khi
  viết claim novelty vẫn phải cập nhật tìm kiếm và đọc toàn văn các đối thủ gần
  nhất.
- Phải so phương trình, dữ liệu, protocol và vai trò lúc inference; không suy
  novelty chỉ từ tên module.

## Các công trình gần nhất

### OneRestore — ECCV 2024

Nguồn:

- Paper: https://arxiv.org/html/2407.04621
- Code: https://github.com/gy65896/OneRestore
- Generator: https://github.com/gy65896/OneRestore/blob/main/syn_data/syn_data.py

Nội dung quan trọng:

- Đề xuất CDD-11 với bốn primitive `low`, `haze`, `rain`, `snow` và 11 loại
  single/composite degradation.
- Imaging order trong paper và code là low → rain → snow → haze, với rain và
  snow không cùng xuất hiện trong CDD-11.
- OneRestore dùng scene descriptor và Scene Descriptor-guided Transformer
  Blocks để điều kiện hóa phục hồi.
- Loss tổng gồm smooth-L1, MS-SSIM và Composite Degradation Restoration Loss.
  CDRL dùng output làm anchor, clean image làm positive, còn input cùng 10 biến
  thể degraded làm negatives trong không gian đặc trưng VGG.
- CDRL không phải supervision trực tiếp cho residual interaction
  `u_ab = r_ab-r_a-r_b`.

Điểm dữ liệu cần nhớ:

- `syn_low` lấy mẫu gamma/noise bằng `numpy.random`.
- `syn_haze` lấy mẫu beta và atmospheric light ngẫu nhiên.
- Rain/snow masks được ghép bằng `random.sample`.
- Generator không ghi manifest và mỗi thư mục degradation được sinh bởi một
  lần chạy riêng. Vì vậy các ảnh cùng scene trong CDD-11 không được bảo đảm dùng
  cùng mask/tham số giữa primitive và composite.
- Muốn có counterfactual tuple hợp lệ phải lấy mẫu một manifest rồi replay nó.

### PRISM — ICLR 2026

Nguồn:

- Proceedings: https://proceedings.iclr.cc/paper_files/paper/2026/hash/0e29b70a7836581919add7f8f622dd2c-Abstract-Conference.html
- Paper HTML: https://arxiv.org/html/2603.14151
- Project: https://prismrestore.github.io/

Nội dung quan trọng:

- Prompted conditional diffusion cho compound và controllable restoration của
  scientific images.
- Huấn luyện trên primitive, mixture tối đa ba distortions, partial prompts và
  negative prompts; thứ tự degradation được lấy ngẫu nhiên.
- Fine-tune CLIP image encoder bằng weighted contrastive loss. Trọng số giữa hai
  degraded variants dựa trên Jaccard overlap của tập distortion:

  `w_jk = exp(1 - |d_j ∩ d_k| / |d_j ∪ d_k|)`.

- Mục tiêu tổ chức latent geometry để mixture liên hệ có cấu trúc với primitive,
  hỗ trợ joint restoration, selective restoration và unseen mixtures.
- Không dùng đúng pixel-space interaction residual `u_ab`, nhưng đã bao phủ
  claim rộng “compound-aware primitive/mixture supervision tạo compositional
  generalization”.

Hệ quả:

- Chỉ thêm primitive/mixture supervision hoặc auxiliary compositional head
  không còn là khác biệt đủ mạnh.
- Một hướng mới phải chỉ ra thông tin/ràng buộc khác PRISM và vai trò thực tế
  của nó trong deployed restoration.

### IMDNet — 2025 preprint

Nguồn: https://arxiv.org/html/2511.04920

Nội dung quan trọng:

- NAFBlock-based encoder–decoder cho multi-degraded restoration.
- Degradation Ingredient Decoupling Block tách clean feature và degradation
  information trong miền spatial/frequency; Task Adaptation Block chọn các
  đường xử lý phù hợp.
- Loss gồm multi-scale Charbonnier, edge, FFT frequency và cosine decoupling
  giữa clean feature `CF` với degradation information `DI`.
- Không thấy target pixel-space `u_ab`, nhưng factor/ingredient decoupling và
  component-adaptive paths đã được nghiên cứu trực tiếp.

### Dual-level Prototype Learning (DPL) — ICCV 2025

Nguồn:
https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Dual-level_Prototype_Learning_for_Composite_Degraded_Image_Restoration_ICCV_2025_paper.html

Nội dung quan trọng:

- Học degradation-level prototypes và factor-level prototypes để biểu diễn cả
  loại suy hao lẫn biến thiên cường độ/thành phần vật lý.
- Scene embeddings từ hai mức prototype điều kiện hóa restoration transformer.
- Prototype Scatter Learning Loss tổ chức phân bố prototype.

Hệ quả: “biểu diễn riêng loại và cường độ/factor của composite degradation” đã
có đối thủ mạnh; một degradation embedding mới cần hơn việc đổi kiến trúc head.

### DFPIR — CVPR 2025

Nguồn:
https://openaccess.thecvf.com/content/CVPR2025/html/Tian_Degradation-Aware_Feature_Perturbation_for_All-in-One_Image_Restoration_CVPR_2025_paper.html

Nội dung quan trọng:

- Đặt bài toán task interference do gradient giữa các degradation xung đột.
- Dùng degradation-aware channel/attention perturbations để điều chỉnh feature
  space cho shared parameter space.

Hệ quả: phát hiện “cross-task interference” hoặc thêm degradation-guided feature
modulation tự nó không phải novelty.

### FAPE-IR — 2026 preprint

Nguồn: https://arxiv.org/abs/2511.14099

Nội dung đã kiểm tra:

- Frequency-aware planning/execution cho all-in-one restoration.
- Dùng frequency-specialized LoRA-MoE trong hệ diffusion/planner để giảm xung
  đột giữa các loại suy hao.

Hệ quả: ghép frequency decomposition, planner và experts không phải khoảng
trống rõ ràng; phiên bản nhẹ hơn chỉ đáng kể nếu chứng minh quality–cost frontier
hoặc cơ chế mới.

### DAME-Net / MDUR — 2026 preprint

Nguồn: https://arxiv.org/abs/2604.09313

Nội dung quan trọng:

- Factor-wise Degradation Perception với multi-label supervision.
- Conditioned Decoupled MoE dùng spatial-frequency processing và routing theo
  từng factor.
- MDUR benchmark có 43 cấu hình từ single đến four-factor composites, kèm
  standardized seen/unseen splits và đánh giá downstream detection.

Hệ quả: factor-wise perception, MoE và unseen/higher-order compositions đã được
kết hợp trong một bài composite restoration gần đây.

### Retrieve-to-Restore (R2R) — CVPR 2026

Nguồn:
https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Retrieve-to-Restore_Efficient_All-in-One_Image_Restoration_with_a_Retrieval-Based_Degradation_Bank_CVPR_2026_paper.html

Nội dung quan trọng:

- Tách degradation adaptation khỏi backbone bằng compact degradation bank.
- Retrieve prior phù hợp để điều biến backbone khi inference.
- Claim chất lượng gần SOTA với khoảng 91% MACs thấp hơn trong thiết lập của
  paper.

Hệ quả: “lightweight retrieval/routing cho AiOIR” đã có đối thủ chính thức ở
CVPR 2026.

### Degradation-Consistent TTA (DCTTA) — CVPR 2026

Nguồn:
https://openaccess.thecvf.com/content/CVPR2026/html/Tang_Degradation-Consistent_Test-Time_Adaptation_for_All-in-One_Image_Restoration_CVPR_2026_paper.html

Nội dung quan trọng:

- Xử lý distribution shift của AiOIR bằng test-time adaptation.
- Sinh re-degradation để tạo pseudo pairs, dùng consistency loss và chỉ cập
  nhật các degradation-sensitive parameters quan trọng.

Hệ quả: test-time adaptation và consistency dưới unseen degradation
distribution đã là một hướng CVPR 2026; không thể dùng “thích nghi khi test” như
một claim chung chung.

### Restore, Assess, Repeat (RAR) — CVPR 2026

Nguồn:
https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Restore_Assess_Repeat_A_Unified_Framework_for_Iterative_Image_Restoration_CVPR_2026_paper.html

Nội dung quan trọng:

- Tích hợp degradation identification, restoration và quality verification
  trong latent space.
- Lặp restore–assess để xử lý unknown/composite degradations.

Hệ quả: một hệ “tự kiểm tra rồi phục hồi tiếp” hoặc quality-aware controller
phải phân biệt rõ với RAR.

### DiTTo — 2026 preprint

Nguồn: https://arxiv.org/abs/2605.30915

Nội dung quan trọng:

- Agent chọn thứ tự gọi các restoration experts.
- Order-aware Restoration Alignment tách degradation identification,
  restoration-action ordering và output format.

Phân biệt cần nhớ:

- DiTTo nghiên cứu **thứ tự hành động phục hồi** trong agent pipeline.
- Audit dự kiến của repo này nghiên cứu **thứ tự hình thành suy hao** và độ bền
  của một one-pass restorer. Hai câu hỏi không đồng nhất, nhưng vẫn cần tìm kiếm
  sâu trước khi tuyên bố novelty.

### OPIR — 2026 preprint

Nguồn: https://arxiv.org/abs/2601.10192

Nội dung quan trọng:

- Dự đoán task-aware inverse degradation operator.
- Sinh uncertainty map để hướng dẫn refinement stage thứ hai.
- Nhấn mạnh efficient AiOIR và restoration reliability.

Hệ quả: uncertainty-guided lightweight restoration cũng không còn là khoảng
trống mặc định.

### ABAIR — 2024 preprint

Nguồn: https://arxiv.org/abs/2411.18412

Nội dung quan trọng:

- Blind AiOIR với per-pixel degradation estimation, independent low-rank
  adapters và adaptive adapter combination.
- Báo cáo generalization sang unseen degradations và composite distortions.

Hệ quả: “adapter nhẹ tổng quát sang unseen/composite” đã có tiền lệ trực tiếp.

### Ingredient-Oriented Multi-Degradation Learning — CVPR 2023

Nguồn:
https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Ingredient-Oriented_Multi-Degradation_Learning_for_Image_Restoration_CVPR_2023_paper.pdf

Nội dung quan trọng:

- Task-oriented prior hubs và ingredient-oriented integration.
- Dùng physics-inspired degradation operations để tạo compositional
  representation.

Hệ quả: ý tưởng phân rã thành “nguyên liệu suy hao” đã có từ trước làn sóng
2025–2026.

### Estimation and Restoration of Compositional Degradation — 2018

Nguồn: https://arxiv.org/abs/1812.09629

Nội dung quan trọng:

- Xem degradation thực tế như chuỗi blur/noise/compression có thứ tự.
- Degradation-estimation CNN dự đoán loại/mức, sau đó restoration CNN nhận các
  thuộc tính đó để phục hồi one-pass.
- Chỉ ra pipeline phục hồi từng bước dễ gây error propagation vì output của
  bước trước không đúng distribution mà bước sau giả định.

Hệ quả: compositional degradation, ước lượng degradation trước restoration và
nhược điểm của sequential restoration không phải phát hiện mới.

## Kết luận từ audit interaction supervision

Với:

```text
r_a  = y - x_a
r_b  = y - x_b
r_ab = y - x_ab
u_ab = r_ab - r_a - r_b,
```

nếu định nghĩa `u_hat = r_ab_hat-r_a-r_b`, thì:

```text
u_hat-u_ab = r_ab_hat-r_ab.
```

Interaction loss trực tiếp chính xác là reconstruction loss. Head `u` độc lập
có gradient mới nhưng chỉ là auxiliary/decomposition supervision; target phụ
thuộc generator và claim rộng va chạm với PRISM. Phiên bản này đã được đóng
trước GPU. Báo cáo chi tiết nằm tại
`results/2026-09-20_interaction_audit/README.md`.

## Giả thuyết chẩn đoán đang mở

CDD-11 dùng một thứ tự sinh cố định, trong khi các phép biến đổi phi tuyến không
giao hoán. Một mô hình có thể đạt điểm tốt bằng cách thích nghi với pipeline đó
nhưng nhạy với thứ tự khác dù tập degradation không đổi.

Câu hỏi chẩn đoán:

> Với cùng ảnh sạch, cùng factor, mask và tham số, A0-L thay đổi hiệu quả phục
> hồi bao nhiêu khi chỉ đảo thứ tự hình thành degradation?

Đây chưa phải contribution. Bằng chứng đầu tiên phải đến từ frozen-model audit,
báo cả input severity, restoration gain và severity-matched control. Chỉ khi có
suy giảm có hệ thống mới tiếp tục rà toàn văn và thiết kế cơ chế order-robust.
