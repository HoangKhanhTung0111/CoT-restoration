# Kế hoạch đối chứng thứ tự tạo suy hao

Cập nhật: 2026-09-20.

## Mục tiêu và phạm vi

Mục tiêu trước mắt không phải tạo module mới, mà kiểm tra một câu hỏi nhân quả hẹp:
việc mô hình nhìn thấy một thứ tự tạo suy hao trong huấn luyện có làm năng lực phục hồi
chuyên biệt theo thứ tự đó hay không.

Thí nghiệm đầu tiên chỉ dùng hai phép suy hao:

- `A = low -> haze`;
- `B = haze -> low`.

Đây là nghiên cứu về **formation order** (thứ tự tạo suy hao), không phải
**restoration-action order** (thứ tự mô hình loại bỏ từng suy hao).

Kết quả audit A0-L ngày 2026-09-20 chỉ là bước phát hiện giả thuyết. Nó dùng năm
clean scene validation và không chứng minh shortcut, nguyên nhân nhân quả, tính mới
hay hiệu quả của một phương pháp mới.

## Protocol bắt buộc

Ba mô hình dùng cùng backbone NAFNet, pretrained initialization, seed, clean-scene
split, số optimizer update, crop, lịch learning rate, loss và quy tắc chọn checkpoint.
Điểm khác duy nhất là phân bố thứ tự tạo suy hao trong dữ liệu huấn luyện:

1. `Fixed-A`: chỉ huấn luyện với `low -> haze`.
2. `Fixed-B`: chỉ huấn luyện với `haze -> low`.
3. `Balanced-order`: cân bằng A/B trong cùng lịch huấn luyện bằng ERM thông thường.

Mọi cặp A/B của cùng scene và realization phải dùng chung tham số, noise,
illumination field, transmission map và atmospheric light. Train và validation phải
tách theo clean scene. Test chính thức không được mở trong giai đoạn phát triển.

Đánh giá chính dùng ảnh raw tạo bởi đúng toán tử trên. Severity matching chỉ được
báo cáo như phân tích phụ vì nó thay đổi ảnh đầu vào; không dùng nó làm dữ liệu huấn
luyện hoặc bằng chứng chính.

Các realization của cùng clean scene không phải mẫu độc lập. Khoảng tin cậy phải
bootstrap theo clean scene sau khi lấy trung bình các realization trong scene.

Metric bắt buộc cho từng thứ tự A/B:

- output PSNR và SSIM;
- PSNR/SSIM gain so với input;
- harm rate;
- trung bình hai order và worst-order;
- specialization difference-in-differences giữa Fixed-A và Fixed-B;
- khoảng cách giữa Balanced-order và specialist tương ứng.

Ngưỡng 0,1/0,2 dB, nếu dùng, chỉ là cổng quản lý pilot được khai báo trước; không
được mô tả là tiêu chuẩn khoa học phổ quát hoặc tiêu chuẩn CVPR.

## Cây quyết định hữu hạn

```text
[Data gate: generator tái lập được, A/B chung tham số, split không rò rỉ]
        |
        +-- Không đạt -> dừng, sửa protocol; không dùng GPU huấn luyện.
        |
        `-- Đạt
              |
              v
       [Run 1 Fixed-A + Run 2 Fixed-B]
              |
              +-- Ưu thế không chạy theo order đã train
              |      -> dừng claim "order shortcut".
              |         Chỉ giữ kết quả như phân tích độ khó/generator.
              |
              `-- Ưu thế đảo chiều theo order đã train
                     -> có bằng chứng order exposure ảnh hưởng năng lực.
                            |
                            v
                    [Run 3 Balanced-order ERM]
                            |
                            +-- Bắt kịp hai specialist
                            |      -> random-order coverage đã đủ;
                            |         không tạo module mới.
                            |
                            `-- Còn kém specialist ở worst-order
                                   -> có dấu hiệu joint-learning limitation.
                                          |
                                          v
                                  [Run 4 Group DRO, có điều kiện]
                                          |
                                          +-- Khắc phục được
                                          |      -> robust objective đã đủ;
                                          |         chưa cần kiến trúc mới.
                                          |
                                          `-- Không khắc phục
                                                 -> dừng GPU; chỉ lúc này mới
                                                    viết cơ chế M, phương trình,
                                                    novelty audit và phép thử bác bỏ.
```

Giai đoạn hiện tại có đúng ba run bắt buộc và tối đa một run có điều kiện. Không
được diễn giải ngân sách bốn run là bảo đảm tạo ra contribution.

## Điều kiện diễn giải

- Evidence cho ảnh hưởng của training order cần hiệu ứng specialization chạy theo
  order đã train và khoảng tin cậy scene-clustered không chứa 0; không chỉ dựa vào
  việc train A rồi test B.
- Nếu Balanced-order bắt kịp specialists, kết luận là augmentation/coverage đã đủ.
  Random-order augmentation riêng nó không phải novelty vì đã có tiền lệ gần.
- Nếu Balanced-order còn kém, specialists chỉ là controls chứ không phải theoretical
  ceilings. Trước một module mới phải có robust-training baseline như group DRO.
- Nếu chỉ dùng CDD-11-30 với năm validation scene, mọi kết luận đều là exploratory
  pilot. Trước claim tổng quát phải mở rộng số clean scene độc lập, khóa test và lặp
  lại trên dữ liệu/protocol phù hợp hơn.

## Bước triển khai hiện tại

1. Tạo dataset synthetic low+haze tái lập được, chung realization giữa A/B và chia
   scene-disjoint.
2. Thêm cấu hình ba run Fixed-A, Fixed-B và Balanced-order.
3. Thêm evaluator chung tạo bảng per-order, scene-cluster bootstrap và các contrast
   đã đăng ký trước.
4. Tạo một Kaggle notebook chạy ba control và đóng gói báo cáo nhẹ.
5. Smoke-test local bằng model compact và ảnh giả; không coi smoke test là kết quả
   nghiên cứu.

## Quyết định sau ba control (2026-09-20)

Ba control đã chạy đúng protocol trên năm clean scene validation. Specialization
difference-in-differences đạt `5,9645 dB`, CI95% scene-clustered
`[5,2890; 6,6399]`. Balanced-order ERM tốt nhất về trung bình và worst-order nhưng
vẫn thấp hơn specialist tương ứng `1,1477 dB`, CI95%
`[0,9416; 1,3399]`. Vì vậy cây quyết định đi tới đúng một Group DRO control.

## Group DRO control được đăng ký trước

- Dữ liệu, seed, pretrained NAFNet-SIDD-width32, 20 epoch, 320 mẫu/epoch, batch,
  optimizer, learning-rate schedule, loss cơ sở và split giữ nguyên như Balanced ERM.
- Hai group là formation order A và B; dữ liệu mỗi epoch cân bằng chính xác 50/50.
- Trọng số group bắt đầu `[0.5, 0.5]`.
- Sau mỗi epoch, tính restoration loss trung bình của từng group trên toàn bộ hai GPU
  rồi cập nhật bằng exponentiated gradient với `eta=0.1`:

  `q_g <- q_g * exp(eta * (L_g - mean(L)))`, sau đó chuẩn hóa tổng về 1.

- Weight decay `0.001` được giữ làm regularization; không sweep eta hay hyperparameter.
- Checkpoint được chọn bằng **worst-order validation PSNR**, thay vì PSNR trung bình.
- Test split tiếp tục bị khóa.

Đối chứng chính dùng chính `per_sample.csv` của ba control trước. Các khóa
`scene_id`, `realization`, `generation_seed`, `order` và input metric phải khớp hoàn
toàn trước khi so sánh.

### Tiêu chí diễn giải đã khóa

- **Khắc phục đầy đủ:** Group DRO cải thiện worst-order so với Balanced ERM, không
  gây giảm trung bình có ý nghĩa, và khoảng cách tới specialist nằm trong margin
  quản lý pilot `0,1 dB`.
- **Khắc phục một phần:** worst-order tăng với CI95% scene-clustered có cận dưới lớn
  hơn 0, nhưng khoảng cách specialist vẫn vượt `0,1 dB`.
- **Không khắc phục:** worst-order không tăng đáng tin cậy, hoặc mức tăng phải đổi
  bằng suy giảm trung bình rõ rệt.

Ngưỡng cải thiện thực dụng `0,2 dB` và equivalence margin `0,1 dB` vẫn chỉ là cổng
quản lý pilot. Dù kết quả thuộc nhánh nào, năm scene và một cặp synthetic low+haze
không đủ cho claim tổng quát hay contribution công bố.
