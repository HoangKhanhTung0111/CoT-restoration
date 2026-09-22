# Kết quả smoke OneRestore và MIRAGE trên CDD-11

Nguồn: `results/2026-09-23_common_fail/common_failure_smoke_results.zip` do user
cung cấp. Notebook chạy tại commit `f0726faf45eb436b0f8f2f6ef15fe5137848a57d`.

## Kiểm tra kỹ thuật

- `run.json`: errors rỗng; cả hai `status.json` là `complete`, mỗi mô hình 9 ảnh.
- Same `manifest_sha256` cho hai model:
  `5ce3bb193111f970b2f4e3a858687fdce1616bea57b2fbfbd8dc8bc5703bd7c4`.
- 195 scene sau exclusion/dedup: discovery 78, confirmation 39, holdout 78.
  Loại 30 ID từng có trong thí nghiệm nội bộ, `near_duplicate_candidates` rỗng
  theo dHash screen. Điều này không chứng minh loại sạch semantic duplicates.
- Checkpoint OneRestore/embedder đúng hash dự kiến; MIRAGE tải và ghi hash
  `6a8216710c80e07af50ee82f96203dc64a3daaddf7c6b6cfc72819a7e6f1ac02`.
- Kaggle PyTorch `2.10.0+cu128`, GPU T4. Mọi output finite và kích thước
  1080×720. Đầu ra raw đôi lúc ngoài [0,1], metric chấm sau clamp như protocol.
- Warm forward trên chính các ảnh này khoảng 0.304 s OneRestore, 1.265 s MIRAGE;
  peak allocated ~2.38 và ~3.48 GiB. Đây là số đo smoke gồm những yếu tố runtime
  của adapter, không thay cho latency paper hay ước lượng toàn bộ download.

## PSNR/SSIM trung bình trên đúng 3 scene

| Suy hao | OneRestore | MIRAGE |
| --- | ---: | ---: |
| Rain | 33.128 / 0.9643 | 34.780 / 0.9723 |
| Low+haze | 25.797 / 0.8350 | 26.866 / 0.8372 |
| Low+haze+snow | 26.230 / 0.8312 | 27.367 / 0.8387 |

MIRAGE cao hơn OneRestore trên cả 9 cặp: chênh PSNR theo rain +1.009,
+2.354, +1.593 dB; low+haze +0.817,+1.269,+1.121;
low+haze+snow +1.055,+1.695,+0.661. Đây là quan sát descriptive trên 3 scene,
không CI, không chứng minh SOTA/khả năng tổng quát hóa.

Đã xem các panel `00370_low_haze_snow`, `00890_rain`,
`00890_low_haze_snow` của cả hai: thứ tự input/output/GT hợp lý, không thấy
hiện tượng màu/kênh đảo rõ rệt. Vẫn có sai khác màu và cấu trúc so với GT ở
composite. Chưa định lượng mất chi tiết hoặc residual artifact theo vùng.

Không so rain với low+haze(+snow) để suy ra nguyên nhân cross-degradation:
ảnh đầu vào có cường độ/tính chất suy hao khác nhau. Cũng không lấy kết quả
3 scene để chọn ngay một loss, module hay claim khoa học.

## Quyết định

**Smoke PASS về kỹ thuật.** Tiếp tục bước đo discovery: đủ 11 degradation trên
78 scene, 2 checkpoint, paired metrics theo scene/group, panel có chọn trước
(scene đầu cho 11 nhóm; 2 scene kế tiếp cho rain và low+haze+snow) để ZIP gọn.
Kiểm tra lỗi lặp lại sau khi có distribution, input PSNR và
ảnh đại diện. Confirmation/holdout không chạy tại bước này.

Notebook: `notebooks/kaggle_common_failure_discovery.ipynb`. Dự tính riêng
inference từ forward smoke ~20 phút nếu tốc độ giữ nguyên; trọn phiên bao gồm
download/extraction/data I/O/metric có thể khoảng 45–90 phút, chưa đo thực tế.
