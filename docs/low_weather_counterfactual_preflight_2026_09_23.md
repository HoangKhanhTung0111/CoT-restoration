# Preflight và phép thử low-light × weather có kiểm soát

Ngày thực hiện: 2026-09-23. Liên hệ với
`docs/common_failure_discovery_result_2026_09_23.md`.

## Kết quả CPU

`python -m unittest hybrid_cot_nafnet.test_low_weather_interaction -v`

6/6 kiểm thử qua: cùng manifest cho bốn view; weather branch không đổi khi
thay mức low; mức low=0 là clean, mức low=1 khớp `apply_factor('low')` cũ;
low luôn đứng trước rain/snow rồi haze; manifest không bị sửa; quantization
8-bit, padding và unpadding cho kích thước lẻ; phép tính contrast và tỷ lệ
sai số có các control identity/perfect. Đây là tensor tổng hợp CPU, không
phải kiểm chứng rằng model frozen đã chạy được ở notebook mới.

Một phản ví dụ số học quan trọng: với tensor 64×64 và manifest seed 20260923,
giá trị `input MSE interaction = E(x_low+weather)-E(x_low)-E(x_weather)+E(y)`
âm mạnh: rain+haze −0.0635/−0.1504 khi low=0.6/1.0; snow+haze
−0.0631/−0.1504. Toán tử haze sau low đã bù lại một phần thay đổi pixel.
Do đó không được lấy dấu của `output MSE interaction` làm bằng chứng trực
tiếp về tương tác lỗi do mạng gây ra. Interaction có thể tồn tại ở generator
ngay với mô hình identity. MSE contrast vẫn được ghi để audit, không dùng
làm tiêu chí GO chính.

## Bộ bốn ảnh và tiêu chí chính

Với clean `y`, cùng manifest `z`, weather `W` có thứ tự `rain→haze` hoặc
`snow→haze`, low operator `L_s` mức s ∈ {0.6,1.0}:

    x00 = y
    x10 = L_s(y)
    x01 = W(y; z)
    x11 = W(L_s(y); z)

`z` được tạo một lần từ chính `y` cho mỗi scene. Rain/snow mask,
haze transmission/atmospheric light và low noise được dùng lại trong tất
cả view; không gọi lại generator sau khi biến đổi nội dung. Ở s=1,
`L_s` đúng low operator cũ; ở s=0.6, suy hao và noise được nội suy về 0
theo thang pixel. Mỗi view được lượng tử hóa 8-bit trước khi vào cả image
encoder và restorer, giống protocol inference CDD-11.

Với mỗi ảnh suy hao, đặt `r = MSE(F(x), y) / MSE(x, y)`.
`r=1` cho mô hình identity và `r=0` cho mô hình lý tưởng. Giá trị `r`
trên 1 nghĩa là tệ hơn input. Tiêu chí mô tả chính là

    D = r(x11) - max(r(x10), r(x01)).

`D>0` nghĩa là tỷ lệ lỗi còn lại ở ảnh chồng suy hao lớn hơn cả hai ảnh
thành phần **trong cùng scene/manifest**. Đo cả full image và inner image
loại 32 pixel biên. Nếu input MSE ≤1e-6 thì `r` là null, không thay bằng
epsilon tùy tiện. Ghi song song input/output MSE từng view và MSE contrast.

Hạn chế: `D>0` chưa chứng minh cross-degradation interference của mô hình,
vì sự chồng suy hao có thể phá thông tin nội dung dù input MSE không tăng.
Các mô hình cũng có thể gặp domain shift do generator kiểm soát khác ảnh
CDD-11 gốc. Đây là screen chẩn đoán cần qua hai model, hai mức low và
generator độc lập trước khi thành claim cơ chế.

## Phạm vi GPU và quy tắc quyết định

Notebook `notebooks/kaggle_low_weather_counterfactual.ipynb` sẽ chạy
OneRestore và MIRAGE frozen. Chỉ dùng 78 scene discovery đã khóa; confirmation
39 scene và holdout 78 scene không mở. Mỗi model có 312 hàng metric
(78 scene ×2 weather ×2 low), với 9 forward độc lập/scene nhờ cache
clean và các view đơn. Hai model tổng cộng khoảng 1.404 forward;
không huấn luyện. Chỉ lưu panel ở scene đầu, sắp xếp input/output/GT.

Đăng ký trước ngưỡng sàng lọc: với từng model × weather × mức low,
trung bình `D` theo 78 scene phải có 95% scene-bootstrap CI hoàn toàn trên 0
và hơn nửa scene có `D>0`; kết quả phải cùng dấu ở `inner32`.
Đây là ngưỡng quyết định nội bộ cho 8 điều kiện, chưa là kiểm định thống kê
đủ cho bài báo. Nếu không đạt ở bất kỳ điều kiện nào:
**NO-GO cho claim lỗi phục hồi riêng do chồng low+weather** theo generator
này. Nếu dương nhất quán: kiểm tra thêm generator thứ hai có tham số/mask
được replay và đối chiếu nguyên nhân loss vùng; chưa huấn luyện mạng mới.
Không chỉnh threshold sau khi xem confirmation. Chưa đặt một ngưỡng dB
“đủ CVPR”, và kết quả dương chỉ tạo giả thuyết có cơ sở hơn.

Nguồn dữ liệu/checkpoint giữ như notebook discovery:
`docs/common_failure_audit_selection_2026_09_22.md`.
Kaggle fresh version tải lại test.zip 3.77 GB và checkpoint; thời gian
ước lượng 45–90 phút, cần đo thực tế. Kết quả cần trả lại:
`low_weather_counterfactual_results.zip`.
