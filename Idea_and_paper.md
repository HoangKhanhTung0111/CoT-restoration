# Ý tưởng nghiên cứu: CoT-NAFNet cho phục hồi ảnh suy giảm tổng hợp

> Tài liệu được hiệu chỉnh ngày 14/09/2026. Tên bài, tác giả, năm và trạng
> thái công bố bên dưới chỉ dùng thông tin có thể kiểm chứng từ trang chính
> thức của hội nghị, IEEE hoặc arXiv. Các kết quả chưa tái lập trong repository
> được ghi là kết quả do tác giả paper báo cáo, không phải kết quả của dự án.

## 1. Bối cảnh và động lực

Trong thực tế, một ảnh có thể đồng thời chịu nhiều suy giảm như thiếu sáng,
sương mù, mưa và tuyết. Khi các suy giảm cùng xuất hiện, tín hiệu của chúng có
thể tương tác và che lấp nhau; vì vậy một mô hình tốt trên suy giảm đơn không
nhất thiết giữ nguyên hiệu quả trên suy giảm tổng hợp.

Các hướng hiện có cho thấy ba đánh đổi chính:

1. Mô hình all-in-one suy luận một lần có hiệu quả cao, nhưng việc nhận biết
   đúng thành phần và mức độ suy giảm vẫn khó.
2. Chain-of-Restoration loại bỏ từng suy giảm theo nhiều bước, giúp tận dụng mô
   hình multi-task đã huấn luyện nhưng tăng thời gian suy luận và có thể tích
   lũy sai số qua các bước.
3. Các mô hình dựa trên Transformer, vision-language model hoặc mô hình sinh có
   thể đạt chất lượng mạnh nhưng thường không phù hợp với mục tiêu CNN nhẹ.

Không nên phát biểu rằng các phương pháp all-in-one “thất bại hoàn toàn”. Các
kết quả của AnyIR, AllRestorer và những công trình gần đây cho thấy hướng này
vẫn rất cạnh tranh. Khoảng trống mà dự án nhắm tới được thu hẹp thành:

> Liệu một CNN NAFNet hiệu quả có thể xử lý suy giảm tổng hợp trong một lần
> forward bằng phân rã suy giảm có giám sát và lập kế hoạch điều chế đa tầng,
> với phần tăng thêm dưới 0,5 triệu tham số hay không?

Đây là giả thuyết thực nghiệm, không phải khẳng định đã được chứng minh.

## 2. Đề xuất CoT-NAFNet

### 2.1 Backbone

Sử dụng NAFNet vì kiến trúc CNN đơn giản, không cần activation phi tuyến thông
thường và có checkpoint công khai cho GoPro/SIDD. Cấu hình chính của dự án là
GoPro-width32, khoảng 17,11 triệu tham số trước khi gắn adapter.

Các checkpoint GoPro và SIDD không có cùng topology. GoPro-width32/64 dùng số
block encoder `[1, 1, 1, 28]`, middle `1`, decoder `[1, 1, 1, 1]`; SIDD-width32/64
dùng `[2, 2, 4, 8]`, middle `12`, decoder `[2, 2, 2, 2]`. Vì vậy phải chọn đúng
preset khi nạp pretrained.

### 2.2 Thinking: biểu diễn nội dung và suy giảm

Từ bottleneck feature, adapter tạo hai embedding:

- `content_embedding`: biểu diễn thông tin mong muốn bất biến khi cùng một scene
  xuất hiện dưới các tổ hợp suy giảm khác nhau;
- `degradation_embedding`: dự đoán vector đa nhãn gồm `low`, `haze`, `rain`,
  `snow`.

Không gọi hai embedding này là “độc lập” hoặc “đã disentangle” nếu chỉ tách bằng
hai projection. Muốn đưa ra tuyên bố đó, quá trình train cần ít nhất một ràng
buộc có thể kiểm chứng, chẳng hạn content-consistency giữa hai degradation view
của cùng scene và phép đo mức tương quan giữa hai không gian.

### 2.3 Planning: sinh kế hoạch điều chế

`degradation_embedding` được biến đổi thành một `plan_embedding`, sau đó sinh
scale/bias theo channel cho bottleneck và từng skip connection. Các đầu gate
được zero-initialize để mô hình hybrid bắt đầu gần như đúng bằng NAFNet đã nạp
pretrained.

Để “planning” không chỉ là tên gọi, cần báo cáo:

- độ chính xác/F1 của dự đoán suy giảm;
- phân bố gate theo từng loại suy giảm;
- can thiệp vào degradation embedding có làm gate thay đổi hợp lý hay không;
- ablation giữa conditioning thông thường và planning head.

### 2.4 Action: phục hồi một lần

Decoder NAFNet nhận bottleneck và skip feature đã được điều chế rồi sinh ảnh
phục hồi trong một forward pass. Điều chế skip connection là cần thiết vì phần
thông tin đi tắt có thể đưa cả chi tiết hữu ích lẫn vết suy giảm sang decoder.

Tên “Thinking → Planning → Action” được dùng như cách tổ chức mô hình. Novelty
không nên dựa riêng vào tên CoT mà phải dựa vào supervision, phép can thiệp và
ablation chứng minh vai trò của từng pha.

## 3. Hàm mất mát dự kiến

Một cấu hình khởi đầu có thể là:

```text
L = L_restore
  + lambda_fft * L_frequency
  + lambda_deg * L_multilabel_degradation
  + lambda_content * L_content_consistency
  + lambda_gate * L_gate_regularization
```

Trong đó:

- `L_restore`: PSNR loss hoặc Charbonnier/L1;
- `L_frequency`: sai khác miền tần số;
- `L_multilabel_degradation`: BCE cho bốn thành phần suy giảm;
- `L_content_consistency`: kéo hai content embedding của cùng scene lại gần;
- `L_gate_regularization`: giữ điều chế ổn định/gần identity khi cần.

Không sao chép Lagrangian loss của CoTIR nếu không tái tạo đúng biến, ràng buộc
và mục tiêu mà công thức đó yêu cầu. Phiên bản pipeline hiện tại mới có
restoration, frequency và degradation loss; content-consistency vẫn là phần
cần triển khai và ablate.

## 4. Dữ liệu và protocol đánh giá

Dataset Kaggle hiện tại là bản nhỏ “CDD-11 30 image pair for each 11
degradation”: 25 scene train và 5 scene test theo cấu trúc đã cung cấp. Mỗi scene
có nhiều degradation variant nên mọi split phải theo scene ID, không chia ngẫu
nhiên từng ảnh.

Bản nhỏ phù hợp để:

- kiểm tra pipeline và memory;
- chọn loss/hyperparameter sơ bộ;
- thực hiện ablation ban đầu.

Nó chưa đủ để tuyên bố khả năng tổng quát ngang với kết quả trên CDD-11 đầy đủ.
Kết quả chính thức nên được xác nhận lại trên bản đầy đủ hoặc ít nhất phải ghi
rõ tên phiên bản, số scene và số ảnh trong mọi bảng.

Protocol tối thiểu:

1. NAFNet baseline và CoT-NAFNet dùng cùng split, crop, seed, pretrained, số
   optimizer step và metric implementation.
2. Chạy ít nhất ba seed khi báo cáo chênh lệch nhỏ.
3. Báo cáo PSNR/SSIM tổng thể và theo từng degradation combination.
4. Báo cáo multi-label F1, exact-match accuracy và gate statistics cho hybrid.
5. Báo cáo số tham số, MACs tại độ phân giải cố định, latency sau warm-up, GPU,
   precision và peak memory.
6. Lưu commit, config, dataset manifest, checkpoint report và log cho mỗi run.

Ablation đề xuất:

1. NAFNet pretrained baseline;
2. bottleneck adapter không skip gate;
3. bottleneck + skip gates;
4. thêm degradation supervision;
5. thêm content-consistency;
6. mô hình đầy đủ với gate regularization nếu thành phần này có ích.

## 5. Mục tiêu có thể kiểm chứng

- Primary: cải thiện PSNR/SSIM so với NAFNet baseline trên cùng CDD-11 protocol.
- Secondary: degradation prediction có F1 tốt và gate có hành vi giải thích
  được qua intervention/ablation.
- Efficiency: adapter dưới 0,5 triệu tham số; cấu hình GoPro-width32 hybrid hiện
  khoảng 17,36 triệu tham số.
- Runtime: đo thay vì đặt trước mốc `<0,1 s/image`. Mọi con số latency phải gắn
  với GPU, độ phân giải, tile size, batch size và FP16/FP32.

Không đặt “vượt AirNet/NAFNet” hoặc “thu hẹp khoảng cách với AllRestorer/M2IR”
thành kết luận trước khi có thí nghiệm. Đây là mục tiêu cần kiểm định.

## 6. Tóm tắt tài liệu liên quan đã kiểm chứng

### NAFNet — ECCV 2022

Chen, Chu, Zhang và Sun đề xuất NAFNet trong *Simple Baselines for Image
Restoration*. Đây là backbone hiệu quả, với checkpoint GoPro và SIDD công khai.
Paper báo cáo 33,69 dB trên GoPro và 40,30 dB trên SIDD cho các cấu hình lớn
tương ứng. Nguồn: [ECVA/ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3043_ECCV_2022_paper.php).

### AirNet — CVPR 2022

AirNet gồm Contrastive-Based Degraded Encoder và Degradation-Guided Restoration
Network, xử lý nhiều loại corruption chưa được chỉ rõ trước. Paper gốc nói mỗi
ảnh trong setting của họ chứa một degradation, vì vậy không nên mặc định rằng
kết quả đó trực tiếp đại diện cho composite CDD-11. Nguồn:
[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2022/html/Li_All-in-One_Image_Restoration_for_Unknown_Corruption_CVPR_2022_paper.html).

### OneRestore — ECCV 2024

OneRestore xây dựng mô hình vật lý cho bốn corruption `low-light`, `haze`, `rain`,
`snow` và đề xuất framework phục hồi composite degradation có scene descriptor.
CDD-11 xuất phát từ công trình này, không phải CoR. Nguồn:
[arXiv:2407.04621](https://arxiv.org/abs/2407.04621).

### Chain-of-Restoration — arXiv 2024

CoR thêm degradation discriminator vào mô hình multi-task pretrained rồi loại
bỏ một degradation basis ở mỗi bước cho đến khi ảnh được phục hồi. Điểm mạnh là
zero-shot composition từ degradation bases; đánh đổi là suy luận nhiều vòng.
Nguồn: [arXiv:2410.08688](https://arxiv.org/abs/2410.08688). Trang arXiv không
xác nhận nhãn “CVPR 2024”, nên tài liệu này không ghi venue đó.

### AnyIR — TMLR 2026

Tên đúng là *Any Image Restoration via Efficient Spatial-Frequency Degradation
Adaptation*. AnyIR dùng joint embedding, gated reweighting và spatial-frequency
parallel fusion; tác giả báo cáo giảm 84% tham số và 80% FLOPs so với baseline
của họ. Bản đầu được nộp arXiv tháng 4/2025 và trang arXiv ghi accepted by TMLR
2026. Nguồn: [arXiv:2504.14249](https://arxiv.org/abs/2504.14249).

### AllRestorer — arXiv 2024

AllRestorer dùng Transformer block mô hình hóa quan hệ giữa degradation và image
embedding, kết hợp image/text descriptor và trọng số thích nghi cho từng
degradation. Tác giả báo cáo tăng 5,00 dB so với baseline của họ trên CDD-11;
con số này chỉ nên so sánh khi protocol trùng khớp. Nguồn:
[arXiv:2411.10708](https://arxiv.org/abs/2411.10708).

### Perceive-IR — IEEE TIP 2025

Tác giả đúng là Xu Zhang, Jiaqi Ma, Guoli Wang, Qian Zhang, Huan Zhang và Lefei
Zhang. Framework học quality prompt nhiều mức và dùng CLIP perception space,
sau đó kết hợp quality perceiver với difficulty-adaptive perceptual loss. Nguồn:
[IEEE Xplore, DOI 10.1109/TIP.2025.3566300](https://ieeexplore.ieee.org/document/10990319/).

### CoTIR — arXiv 2026

CoTIR internalize CoT-style reasoning vào mục tiêu học khả vi lấy cảm hứng từ
Lagrangian optimization, dựa trên một mô hình image-editing pretrained và giới
thiệu CoTIR-Bench 5,2 triệu sample. Đây không phải adapter CNN nhỏ và cũng không
nên giản lược thành ba convolution “thinking/planning/action”. Nguồn:
[arXiv:2606.17557](https://arxiv.org/abs/2606.17557). Nguồn này chưa ghi TPAMI.

### M2IR — arXiv 2026

M2IR kết hợp Mamba-Style Transformer để điều tiết degradation propagation ở
encoder và mixture-of-experts được DA-CLIP router hướng dẫn ở decoder. Nguồn:
[arXiv:2603.14816](https://arxiv.org/abs/2603.14816). Nguồn này chưa ghi TPAMI,
vì vậy không gán venue đó khi chưa có DOI/trang nhà xuất bản.

### DPC-Net — arXiv 2026

Tên đúng là *DPC-Net: Dual-Prior Collaborative Network for All-in-One Image
Restoration*. Mô hình dùng degradation-semantic coupled prior được VLM giám sát
và low-level visual prior trong reconstruction. Tác giả là Zhaokun He cùng cộng
sự. Nguồn: [arXiv:2608.20141](https://arxiv.org/abs/2608.20141). Bản nguồn hiện
không ghi AAAI, nên không gán venue AAAI 2026.

### Referring Flexible Image Restoration — arXiv 2024

RFIR đặt bài toán chỉ loại bỏ degradation được chỉ định bởi text trong ảnh có
nhiều suy giảm. Dataset RFIR có 153.423 sample và năm degradation cơ bản; mô
hình TransRFIR sử dụng attention với agent token. Đây là bài toán controllable
restoration, liên quan nhưng khác mục tiêu tự động loại bỏ toàn bộ suy giảm của
dự án. Nguồn: [arXiv:2404.10342](https://arxiv.org/abs/2404.10342).

### Survey về All-in-One Image Restoration — IEEE TPAMI 2025

Survey của Jiang và cộng sự hệ thống hóa taxonomy, protocol và hướng nghiên cứu
AiOIR. Trang arXiv liên kết DOI `10.1109/TPAMI.2025.3598132`, vì vậy trạng thái
TPAMI 2025 có thể kiểm chứng. Nguồn:
[arXiv:2410.15067](https://arxiv.org/abs/2410.15067).

## 7. Định vị đóng góp dự kiến

Đóng góp có khả năng bảo vệ tốt nhất không phải là “đưa CoT vào CNN” theo nghĩa
tên gọi, mà là tổ hợp có thể đo lường:

1. NAFNet single-pass với overhead dưới 0,5M tham số;
2. decomposition content/degradation được giám sát trên paired multi-view scene;
3. degradation-conditioned plan điều khiển bottleneck và mọi skip scale;
4. protocol ablation và intervention chứng minh mỗi thành phần có tác dụng;
5. kết quả được tái lập bằng checkpoint, manifest, commit và log đầy đủ.

Nếu content-consistency, planning intervention hoặc lợi ích trên baseline không
được chứng minh, tên mô hình nên được mô tả thận trọng là
“degradation-aware gated NAFNet” thay vì tuyên bố internalized chain-of-thought.
