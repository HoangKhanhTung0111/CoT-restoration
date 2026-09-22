# Notebook smoke OneRestore + MIRAGE

Notebook: `notebooks/kaggle_common_failure_smoke.ipynb`.

## Cách dùng

Import notebook từ GitHub như các lần trước. Bật Internet và GPU T4, Run All.
Không cần upload kết quả cũ/dataset nhỏ; không có cờ false cần đổi thành true.
Tự tải test CDD-11 đầy đủ (~3.77 GB ZIP), hai checkpoint OneRestore (~73 MB),
checkpoint MIRAGE (~112 MB theo tác giả) và code/phụ thuộc. Cần thêm dung lượng
giải nén; chưa đo chính xác thời gian tải hoặc inference trên Kaggle.

Mỗi model chạy tiến trình riêng, tuần tự trên GPU 0. Đây là inference batch 1,
không phải training DDP; GPU thứ hai không bắt buộc sử dụng. Tổng mặc định
18 lượt inference: 3 scene discovery × 3 degradation × 2 model. Chưa full audit.

Gửi lại `/kaggle/working/common_failure_smoke_results.zip`, kể cả khi model báo
lỗi. ZIP gồm logs, manifest, hash checkpoint, environment, trạng thái, metric
và panel input/output/GT. Không đưa checkpoint/dataset lớn vào ZIP.

## Những gì được khóa

- Hai repo tác giả pin commit; các file OneRestore và test ZIP kiểm tra SHA256
  từ metadata nguồn. MIRAGE ghi SHA256 thực tải (chưa có expected hash độc lập).
- Scene cũ lấy từ manifest baseline đã version trong project; loại theo ID và
  kiểm tra trùng pixel/dHash khi scene cũ có trong archive mới.
- dHash Hamming <=4 là sàng lọc bảo thủ, không chứng minh hết near-duplicates.
  Ảnh cũ không có trong archive mới chưa được so pixel. Không gọi dữ liệu là
  hoàn toàn độc lập với lịch sử các tác giả.
- Thứ tự hash cố định, discovery/confirmation/holdout =40/20/40 theo scene.
  Chỉ discovery được inference. Metadata/fingerprint clean dùng để chia tập,
  không xem output giữ kín. Không được dùng official test toàn bộ như test
  độc lập nếu đã dùng phần discovery để phát triển phương pháp.
- Full-resolution, replicate padding đến bội 8, bỏ padding khi chấm. RGB FP32,
  output clamp [0,1], PSNR và SSIM skimage cùng implementation cho hai model.
  Đây là protocol chung kiểm tra kỹ thuật, không bảo đảm tái hiện bảng paper.
- Checkpoint `weights_only=True`, strict state_dict; lỗi format sẽ dừng thay vì
  dùng random weights hoặc tự bật unpickling không giới hạn.
- OneRestore dùng image encoder tự chọn degradation, không nhãn GT. Chỉ embedder
  resize PIL 224x224 theo code gốc. Constructor GloVe lấy tensor từ checkpoint,
  ResNet khởi tạo không tải ImageNet rồi nạp strict toàn bộ checkpoint. Không
  thay công thức forward hoặc trọng số pretrained.
- MIRAGE bỏ wrapper Lightning chỉ để nạp state_dict tiền tố `net.` vào cùng
  class MIRAGE. Tuple output dùng phần ảnh đầu tiên như script tác giả.

## Kiểm tra local đã thực hiện

`python -m unittest hybrid_cot_nafnet.test_common_failure_audit -v`

Ba test đạt: manifest tái lập/exclusion/partition/missing pair; ZIP traversal;
compile các code cell notebook. Không tải 3.77 GB về máy local, chưa nạp
checkpoint thực hoặc chạy hai mạng trên GPU. Không gọi notebook là đã xác minh
end-to-end. Smoke Kaggle là bước xác minh đó, và có thể phát hiện lỗi tương thích.

## Điều kiện đi tiếp

Hai trạng thái complete chỉ xác nhận model chạy và đầu ra hợp lệ về kỹ thuật.
Phải xem panel/metric/VRAM trước khi chốt cấu hình đánh giá rộng. Timing chứa
cold-start, không dùng làm latency paper. Smoke không chứng minh lỗi chung,
không phải contribution và không mở lại response weighting đã NO-GO.

Nguồn code đã đọc: https://github.com/gy65896/OneRestore và
https://github.com/Amazingren/MIRAGE. Lựa chọn và rủi ro protocol được ghi trong
`docs/common_failure_audit_selection_2026_09_22.md`. OmniRestore chưa đưa vào
notebook do cần xác minh riêng khác biệt resize/normalization.
