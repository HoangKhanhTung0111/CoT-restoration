# Discovery audit: OneRestore và MIRAGE trên 78 scene CDD-11

Nguồn duy nhất cho số liệu dưới đây:
`results/2026-09-23_common_fail2/common_failure_discovery_results.zip` do user
cung cấp, notebook tại commit `cef441d9f3975779ce62556cf9a793464d12134d`.
Chỉ phần discovery đã mở; confirmation 39 scene và holdout 78 scene chưa đánh giá.

## Kiểm tra kết quả

- `run.json`: không lỗi. Cả hai model `complete`, mỗi model 858 ảnh = 78 scene ×
  11 degradation. Mỗi loại đủ 78 scene, hai model dùng cùng manifest SHA256.
- Cùng 195 ID và pixel SHA256 ảnh sạch với smoke trước. Chín ảnh smoke trùng
  chính xác PSNR/SSIM trong run discovery; không có lệch phiên bản dữ liệu.
- Checkpoint hash giống smoke trước. Protocol chung FP32 RGB, clamp output [0,1]
  trước metric, pad rồi bỏ pad; cả hai chạy GPU T4. Đây là phép đo discovery,
  không phải reproduction đúng toàn bộ protocol của từng paper.
- Số liệu theo scene, không coi 858 ảnh của cùng 78 scene là 858 scene độc lập.

## Bảng đầy đủ (PSNR/SSIM trung bình trên 78 scene mỗi nhóm)

| Degradation | Input PSNR | OneRestore | MIRAGE |
| --- | ---: | ---: | ---: |
| low | 12.17 | 26.39 / .8367 | 27.70 / .8447 |
| haze | 15.87 | 32.49 / .9875 | 32.40 / .9890 |
| rain | 22.42 | 33.77 / .9658 | 35.09 / .9731 |
| snow | 24.42 | 35.12 / .9736 | 37.23 / .9806 |
| low+haze | 14.99 | 25.27 / .8214 | 26.68 / .8313 |
| low+rain | 14.65 | 25.33 / .8072 | 26.72 / .8217 |
| low+snow | 13.44 | 24.74 / .7853 | 26.47 / .8049 |
| haze+rain | 14.86 | 29.76 / .9556 | 30.00 / .9660 |
| haze+snow | 14.94 | 29.89 / .9619 | 29.57 / .9669 |
| low+haze+rain | 15.36 | 24.48 / .7898 | 25.63 / .8019 |
| low+haze+snow | 14.80 | 24.83 / .7946 | 26.03 / .8046 |

Macro trung bình trên 11 nhóm: OneRestore 28.370 dB/.8799; MIRAGE
29.410 dB/.8895. MIRAGE hơn 1.041 dB trung bình nhưng không thắng đều:
OneRestore nhỉnh hơn về PSNR ở haze và haze+snow. Không dùng một trung bình
này làm ranking paper.

Restoration PSNR thấp hơn chính input trên 13/858 ảnh OneRestore, 0/858
ảnh MIRAGE. Không có ảnh nào cả hai cùng gây hại theo tiêu chí này; do đó
"cả hai thường làm ảnh tệ hơn input" là claim sai. Một số outlier OneRestore
rất mạnh (scene 07189, snow −12.06 dB và haze −7.14 dB), cần xem riêng nếu
nghi lỗi inference nhưng không thể xem output vì run chỉ lưu panel đã chốt trước.

## Dấu hiệu chung đáng kiểm tra

Trên cùng scene, so `low+haze+rain` với `haze+rain`:

| Model | Chênh PSNR đầu ra | Số scene thấp hơn | 95% bootstrap CI theo scene |
| --- | ---: | ---: | ---: |
| OneRestore | −5.28 dB | 67/78 | [−6.28, −4.21] |
| MIRAGE | −4.38 dB | 68/78 | [−5.33, −3.44] |

PSNR đầu vào của nhóm thứ nhất trung bình lại **cao hơn 0.49 dB**; 46/78
scene có input PSNR cao hơn. Trong số đó, 39 scene OneRestore và 38 scene
MIRAGE vẫn cho output PSNR thấp hơn.

So `low+haze+snow` với `haze+snow`: input delta −0.14 dB;
output delta −5.06 dB OneRestore (69/78 thấp hơn), −3.54 dB MIRAGE
(62/78 thấp hơn). So hai nhóm triple với `low` đơn, input PSNR cao hơn
3.19/2.63 dB, nhưng output thấp hơn 1.91/1.56 dB OneRestore và
2.07/1.67 dB MIRAGE. Đây là bằng chứng mô tả rằng PSNR đầu vào tổng thể
không giải thích hết thứ tự chất lượng đầu ra.

Các con số KHÔNG chứng minh hiệu ứng tương tác nhân quả: CDD-11 các folder
degradation được sinh ở những lượt ngẫu nhiên khác nhau; `haze+rain` và
`low+haze+rain` của cùng scene không chắc chia sẻ mask và tham số. Input
PSNR bằng nhau cũng không có nghĩa giữ lượng thông tin nội dung bằng nhau.
Các bootstrap CI mô tả biến thiên theo scene trong tập discovery, không
chuyển thành claim tổng quát ngoài generator này.

Panel đã xem: scene 03266 ở `haze+rain`, `low+haze+rain`, `haze+snow`,
`low+haze+snow`, `low`; scene 00370 và 00890 ở smoke. Không thấy đảo kênh
hoặc lệch kích thước. Trên các ảnh low-composite, vùng cây/núi tối và tương
phản cục bộ có sai khác so với GT ở cả hai model, nhưng chưa có metric vùng
hoặc annotation đủ để claim "mất chi tiết" là lỗi thống trị.

## Giả thuyết kế tiếp, chưa phải contribution

**H0:** Chênh lệch này chủ yếu do lượng thông tin bị mất/đặc tính riêng của
generator, cộng với lỗi low-light đơn; không có phần hỏng thêm riêng khi low
chồng lên haze/weather sau khi kiểm soát dữ liệu và độ nặng.

**H1:** Khi cố định ảnh sạch và đúng cùng hiện thực hóa haze/rain/snow, bật
low-light tạo thêm sai số phục hồi vượt phần low-light và weather riêng; hiệu
ứng xuất hiện ở cả hai model và qua generator thứ hai.

Thử nghiệm quyết định: tạo bộ bốn trên cùng clean scene và cùng weather
manifest: clean, low, weather, low+weather. Giữ thứ tự/cường độ rõ ràng,
thực hiện thêm các mức severity. Chấm MSE theo scene/vùng và contrast
`E(low+weather)-E(low)-E(weather)+E(clean)`; tránh dùng PSNR trực tiếp trong
contrast cộng trừ vì PSNR logarithmic. So input MSE/độ tối/độ che khuất
và examine representative panels; khóa bộ confirmation trước khi mở.
Nếu contrast không ổn định theo model, severity và generator, dừng claim
"coupling". Nếu ổn định, đây mới là giới hạn cụ thể để đối chiếu với cơ chế
gần nhất; chưa đủ tự nó cho CVPR. Không chọn module mới chỉ từ bảng này.

Nguồn đối chiếu đã có: OneRestore ECCV 2024 và MIRAGE ICLR 2026 đã trực tiếp
huấn luyện composite CDD-11; GenDeg CVPR 2025 khảo sát data synthesis cho
AiOIR. Chỉ chứng minh cùng task còn lỗi, chưa chứng minh khoảng trống novelty.
https://github.com/gy65896/OneRestore
https://github.com/Amazingren/MIRAGE
https://openaccess.thecvf.com/content/CVPR2025/html/Rajagopalan_GenDeg_Diffusion-based_Degradation_Synthesis_for_Generalizable_All-In-One_Image_Restoration_CVPR_2025_paper.html

**Quyết định:** Bước đo lỗi chung đạt mục tiêu mô tả. Chưa mở confirmation hay
training. Bước kế tiếp là thiết kế kiểm soát counterfactual trên generator;
phải kiểm tra replay từng factor và algebra của contrast trên CPU trước GPU.
