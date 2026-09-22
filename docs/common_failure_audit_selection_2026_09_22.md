# Bước 1 — Chọn mô hình/dữ liệu khảo sát lỗi chung

Ngày kiểm tra: 2026-09-22. Trạng thái: đã kiểm tra nguồn và code; chưa tải
checkpoint/dataset, chưa thực thi inference. Không huấn luyện hoặc push GitHub.

## Quyết định

Giữ trọng tâm composite AiOIR. Chọn **OneRestore CDD-11 + MIRAGE CDD11 Small**
làm hai mô hình chính. **OmniRestore image-only** là đối chứng thứ ba có điều kiện
sau kiểm tra preprocessing. Đây là khảo sát lỗi, không phải bảng xếp hạng SOTA.
A0-L chỉ là tham chiếu nội bộ, không ngang bằng ngân sách/dữ liệu với mô hình công khai.

Không lấy PromptIR/AdaIR checkpoint ba/năm tác vụ đơn làm đối thủ composite mặc
định: lỗi trên composite có thể đơn giản do ngoài phân phối huấn luyện. Không
chuyển sang noise/rain/haze đơn chỉ vì thuận tiện tải checkpoint.

## Mô hình và nguồn cố định

### OneRestore — chính

- Repo: https://github.com/gy65896/OneRestore
- Git tree kiểm tra: `23a2e6af066864dca013aeed7f6c6f5958aa7259`.
- Nguồn trọng số: https://huggingface.co/gy65896/OneRestore/tree/main
- `onerestore_cdd-11.tar`: 23,993,607 bytes; LFS SHA256
  `e02e2d87ce56740a9bedb5cffe8b129d3b206590ae3b97050c9b53aa549c7c9f`.
- `embedder_model.tar`: 48,638,120 bytes; LFS SHA256
  `01c1ed1fe4fd06a73c78a7d1fac1b4092a098f3ea08bc71bcab1e4fdad7f6ec2`.
- Dùng blind/image-only, không cấp prompt ground-truth degradation.
- `test.py` resize ảnh cho embedder về 224x224, không được suy ra rằng toàn bộ
  restoration cũng resize. Kiểm tra nhánh restorer khi viết adapter.
- Repo chỉ dẫn môi trường cũ; không hạ PyTorch toàn workspace theo README.

### MIRAGE CDD11 Small — chính

- Repo: https://github.com/Amazingren/MIRAGE
- Git tree: `926dc83646272a0d99b36f4bff496985ab064795`.
- Checkpoint tác giả: https://drive.google.com/file/d/1GLRMUDfjgWR7aW4DqnUDJzsPO98-cw5G/view
- Trang tải hiển thị tên `CDD11_small.ckpt`; README ghi 111.8M dung lượng file,
  10M params. Đây là số tác giả, chưa đo tại project.
- README ghi CDD11 inference TODO nhưng `test_small.py` có `CDD11Tester`,
  `--trainset CDD11_all`, `--cdd11_path`, và load `net.mirage_small.MIRAGE`.
- Có đường code thực thi; chưa xác minh state_dict khớp checkpoint hoặc tải
  binary không bị quota Drive. Không đánh dấu ready-to-run trước smoke test.

### OmniRestore — thứ ba có điều kiện

- Repo: https://github.com/Judith989/omnirestore
- Git tree: `868e19add8d9b1aa8f1f0d17a1b831459216e042`.
- GitHub tree chứa `ckpts/best.ckpt` 37,865,053 bytes và
  `logs/embedder_resnet18_bs64_optadamw_lr1.00e-05_cw1.0_conw0.1_temp0.07_20260101-190539.pt`
  65,845,563 bytes. Binary chưa tải/deserialize.
- Chọn `test_only_images.py`, không `test_with_text.py`.
- `utils/dataset_loader.py` resize input VÀ target; mặc định test 224x224,
  ImageNet normalization. Phải tách protocol reproduction của tác giả và
  protocol độ phân giải chung. Không so PSNR resize với PSNR full-resolution.
- Manifest công khai có train 11,609 dòng/1,183 clean paths; val 1,289 dòng/
  818 clean paths. Cả 818 val clean paths nằm trong train. Đây là overlap
  train–val theo scene, không phải bằng chứng test leakage hoặc cáo buộc kết quả sai.
- Công trình CVPR Workshops 2026, không gọi là CVPR main conference.

## Dữ liệu khảo sát

Nguồn gốc: https://huggingface.co/datasets/gy65896/CDD-11/tree/main
Metadata kiểm tra qua API `/api/datasets/gy65896/CDD-11/tree/main`:

- `test.zip`: 3,770,337,157 bytes (~3.77 GB decimal), SHA256 theo LFS:
  `421567fceda89659a6d047fc78d82f77c68703203336d9c795213ba7096646dc`.
- `train.zip`: 22,403,961,860 bytes (~22.40 GB). Chưa cần tải để inference.
- Manifest test công khai của OmniRestore có 200 clean paths, 2,400 dòng
  (gồm clear). Phải kiểm kê archive thực tế; không lấy số manifest thay kiểm
  tra đủ cặp, unique scene và dedup ảnh sau tải.

**“Mới” ở đây là chưa dùng trong các thử nghiệm nội bộ, không phải benchmark mới
hoặc dữ liệu chắc chắn chưa từng được tác giả dùng để chọn checkpoint.** Không
tuyên bố checkpoint chỉ thấy training nếu chưa xác minh được provenance.

Trước khi mở ảnh hoặc tính output:

1. Kiểm kê tên và hash clean ảnh, loại scene trùng bộ nhỏ cũ; kiểm tra near-duplicate.
2. Phân chia scene còn lại bằng thứ tự hash SHA256("audit-20260922:" + scene_id):
   40% discovery, 20% confirmation, phần còn lại sealed holdout; ghi manifest cố định.
   Mọi degradation của một clean scene đi cùng partition, không chia theo crop.
3. Chỉ dùng discovery để tìm lỗi; confirmation mở sau khi khóa một giả thuyết.
   Không xem output holdout trong khảo sát này.
4. Vì đang dùng một phần official test để phát triển, không được sau đó báo
   toàn bộ official test là đánh giá độc lập của phương pháp mới. Holdout còn lại
   chỉ độc lập với lựa chọn nội bộ, không bảo đảm độc lập với lịch sử tác giả.
5. Dataset/generator bên ngoài để xác nhận tổng quát hóa CHƯA được chốt; bắt buộc
   chọn và kiểm tra overlap trước claim phương pháp. CDD-11 một mình chưa đủ.

Không trộn clear vào macro-average 11 degradation. Có thể đánh giá clear như
identity diagnostic riêng (mô hình có làm hỏng ảnh sạch không).

## Cửa kiểm tra tiếp theo — trước chạy hàng loạt

- Tải checkpoint từ nguồn trên, ghi SHA256 và revision thực tế. Không thực thi
  code từ checkpoint; ưu tiên weights-only, kiểm tra format thay vì tự fallback
  unpickling tùy ý. Dùng môi trường riêng nếu phụ thuộc khác nhau.
- Strict state_dict và RGB/range/normalization/output-shape smoke test trước.
- Hai model chính phải chạy thành công trên cùng 3–5 scene discovery, đủ single/
  double/triple. OneRestore và MIRAGE là tối thiểu; không trì hoãn vô hạn vì model thứ ba.
- Dùng metric implementation chung: PSNR RGB [0,1], SSIM cùng cấu hình,
  pad/crop được ghi rõ và bỏ pad trước chấm điểm. Báo macro theo scene/group.
- Chạy FP32 đầu tiên; nếu dùng AMP phải kiểm tra sai lệch. Không ép một GPU mỗi
  model nếu vượt VRAM; batch 1, đo thời gian và peak memory, không hứa giờ Kaggle.
- OmniRestore giữ riêng kết quả official resize; chỉ thêm vào bảng chung khi
  adapter resolution hợp lệ được xác minh, không resize GT của riêng một model.
- Kiểm tra panel input/output/GT để phát hiện lỗi preprocessing trước khi gọi
  artifact là failure mode nghiên cứu. Không coi cross-model failure là bằng
  chứng nguyên nhân interference khi chưa có đối chứng.

## Đầu ra bước 3 được khóa trước

Mỗi ảnh: scene, degradation, checkpoint SHA, protocol, PSNR, SSIM, runtime;
panel cùng vùng crop cho mọi model, lấy mẫu cả tốt/trung bình/xấu. Nhãn lỗi
khảo sát: mất chi tiết, artifact còn lại, lệch màu/sáng, lỗi biên. Cho phép
"không rõ" và không ép mọi lỗi vào một giả thuyết đã chọn trước.

Đi tiếp chỉ khi có lỗi lặp ở ít nhất hai mô hình phù hợp cùng task và tồn tại
trên nhiều scene, không phải riêng preprocessing hoặc một generator artifact.
Tần suất/độ lớn cần báo đầy đủ, không chỉ cherry-pick ảnh. Lúc đó mới đối chiếu
literature và thiết kế cơ chế, không huấn luyện loss mới ngay.

## Handoff

Bước chọn nguồn đã xong; bước xác minh checkpoint bằng thực thi còn chưa làm.
Chưa tải 3.77 GB hoặc các trọng số. Việc tiếp theo: chuẩn bị adapter và download/
manifest trong notebook Kaggle, chạy smoke trước rồi đo ngân sách inference.
Giữ workflow GitHub -> Kaggle import; chưa cần user bật Kaggle trong bước chọn này.
