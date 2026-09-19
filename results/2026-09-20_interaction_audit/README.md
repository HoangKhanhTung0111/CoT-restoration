# Kiểm toán trước GPU: interaction supervision

Ngày kiểm toán: 2026-09-20.

## Kết luận

**NO-GO đối với đề xuất hiện tại; chưa tạo notebook và chưa dùng GPU.**

Đại lượng tương tác vẫn là một phép đo hợp lệ của generator, nhưng chưa tạo
thành một phương pháp nghiên cứu đủ chặt:

1. loss trực tiếp nhất chính xác là reconstruction loss viết lại;
2. head tương tác độc lập tạo tín hiệu gradient mới, nhưng nếu không tham gia
   đầu ra thì chỉ là auxiliary supervision;
3. nếu cho các head thành phần và tương tác cộng thành đầu ra, mục tiêu không
   còn tương đương reconstruction, nhưng decomposition là do generator quy
   định và chưa được nhận dạng duy nhất từ ảnh composite;
4. CDD-11 hiện có không lưu cùng mask và tham số ngẫu nhiên giữa các biến thể;
5. PRISM (ICLR 2026) đã bao phủ claim rộng về compound-aware supervision và
   cấu trúc compositional giữa primitive/mixture.

Vì vậy không được tuyên bố rằng interaction supervision hiện tại là
contribution mới, và không đáng tiêu GPU trước khi giải quyết đồng thời tính
hợp lệ của target lẫn khoảng trống so với PRISM.

## 1. Đặc tả

Từ ảnh sạch `y`, đặt:

```text
x_a  = D_a(y)
x_b  = D_b(y)
x_ab = D_b(D_a(y))

r_a  = y - x_a
r_b  = y - x_b
r_ab = y - x_ab
u_ab = r_ab - r_a - r_b
```

Có thể viết lại:

```text
u_ab = [D_b(y) - y] - [D_b(D_a(y)) - D_a(y)].
```

Do đó `u_ab` là **sai phân có hướng**: tác động của `D_b` thay đổi bao nhiêu
khi đầu vào đã qua `D_a`. Nó có ý nghĩa đối với pipeline sinh đã chọn, nhưng
không phải thành phần vật lý độc lập, không đối xứng theo `a,b`, và có thể đổi
khi đổi thứ tự, clipping hoặc tham số.

## 2. Rút gọn loss

### 2.1. Dùng residual cuối để định nghĩa interaction dự đoán

Nếu:

```text
u_hat = r_ab_hat - r_a - r_b,
```

thì với mọi chuẩn sai số dựa trên hiệu:

```text
u_hat - u_ab
= r_ab_hat - r_a - r_b - (r_ab - r_a - r_b)
= r_ab_hat - r_ab.
```

Vì vậy interaction loss và composite reconstruction loss có cùng giá trị và
cùng gradient. Cấu hình này bị loại.

### 2.2. Head interaction độc lập

Với `U_theta(x_ab)` và loss `||U_theta(x_ab)-u_ab||`, loss không tương đương
reconstruction: nó truyền gradient vào một head độc lập. Tuy nhiên:

- nếu đầu ra phục hồi không dùng head này, đây chỉ là auxiliary task;
- muốn quy lợi ích cho interaction, phải thắng một auxiliary head có số tham
  số và cường độ loss tương đương;
- tên gọi interaction không tự làm auxiliary target trở thành cơ chế mới.

### 2.3. Decomposition tham gia đầu ra

Cấu hình không tương đương reconstruction tối thiểu là:

```text
r_ab_hat(x_ab) = A_theta(x_ab) + B_theta(x_ab) + U_theta(x_ab),
```

với supervision riêng cho `r_a`, `r_b`, `u_ab`. Reconstruction chỉ ràng buộc
tổng ba head; component loss chọn một decomposition cụ thể. Đây là tín hiệu
huấn luyện mới về mặt toán học.

Nhưng target `r_a` và `r_b` được đo trên `x_a` và `x_b`, không phải các thành
phần quan sát trực tiếp trong `x_ab`. `u_ab` hấp thụ cả thứ tự, clipping và mọi
khác biệt của generator. Vì thế cấu hình này chưa vượt qua cổng ý nghĩa và
identifiability.

### 2.4. Consistency loss

Ràng buộc:

```text
||r_ab_hat - r_a_hat - r_b_hat - u_ab_hat||
```

có thể đổi gradient trong quá trình tối ưu, nhưng bằng 0 tự động khi bốn head
đạt các target nói trên. Nó không thêm nhãn hay thông tin nhận dạng mới; phải
được xem là regularizer, không phải contribution độc lập.

## 3. Kiểm tra gradient số

Chạy CPU:

```powershell
python -m hybrid_cot_nafnet.audit_interaction_losses
```

Kết quả đầy đủ nằm trong `audit.json`. Các điểm chính:

| Kiểm tra | Kết quả |
|---|---:|
| Chênh lệch loss giữa naive interaction và reconstruction | `0.0` |
| Chênh lệch gradient lớn nhất | `1.39e-17` |
| Gradient reconstruction lên independent interaction head | `0.0` |
| Gradient interaction loss lên head đó | `0.1542` |
| Consistency loss tại đúng component targets | `1.35e-33` |

Sai số cỡ `1e-17` là sai số số học float64, xác nhận kết quả đại số.

## 4. Kiểm toán dữ liệu

Loader hiện tại chỉ ghép ảnh bằng `scene_id` và tên loại suy hao. Nó không có
manifest chứa rain/snow mask, gamma/noise của low-light, beta/atmospheric light
của haze, hay thứ tự phép biến đổi.

Mã sinh chính thức của OneRestore cũng lấy mẫu ngẫu nhiên độc lập bằng
`numpy.random` và `random.sample`. Mỗi lần gọi sinh một thư mục degradation có
thể dùng mask và tham số khác. Vì vậy các thư mục CDD-11 đã sinh không bảo đảm
tuple `(y, x_a, x_b, x_ab)` dùng cùng realization.

Muốn tạo target tương tác hợp lệ phải sửa generator để:

1. lấy mẫu một manifest theo scene;
2. replay đúng manifest cho primitive và composite;
3. lưu thứ tự biến đổi cùng mọi tham số/mask;
4. tách scene trước khi sinh train/validation/test.

Việc này khả thi về kỹ thuật, nhưng chưa nên thực hiện trước khi có phương pháp
và claim mới vượt qua kiểm toán literature.

## 5. Đối chiếu literature gần nhất

| Công trình | Phần đã có | Hệ quả |
|---|---|---|
| OneRestore (ECCV 2024) | CDD-11, composite synthesis và contrastive loss dùng nhiều degraded negatives | Dùng nhiều biến thể cùng scene để supervision không mới; chưa thấy target `u_ab` trực tiếp. |
| IMDNet (2025) | Spatial/frequency ingredient decoupling, component-adaptive paths và cosine decoupling loss | Claim factor/component supervision đã đông; chỉ thêm head thành phần là yếu. |
| PRISM (ICLR 2026) | Compound-aware supervision, primitive/mixture variants và weighted contrastive compositional latent geometry | Trùng mạnh với claim rộng “structured supervision giúp generalize sang mixture”. |
| Uchida et al. (2018) | Ước lượng loại/mức degradation và one-pass restoration cho degradation mắc nối tiếp | Premise rằng thứ tự làm thay đổi degradation và pipeline tuần tự dễ lỗi đã có từ lâu. |

Nguồn chính:

- OneRestore paper: https://arxiv.org/html/2407.04621
- OneRestore generator: https://github.com/gy65896/OneRestore/blob/main/syn_data/syn_data.py
- IMDNet: https://arxiv.org/html/2511.04920
- PRISM: https://arxiv.org/html/2603.14151
- Compositional degradation (2018): https://arxiv.org/abs/1812.09629

Không tìm thấy trong các nguồn đã kiểm tra một loss pixel-space đúng bằng
`u_ab`, nhưng điều đó **không đủ** để bảo vệ novelty: PRISM đã chiếm phần claim
khái niệm quan trọng hơn, còn `u_ab` hiện chưa chứng minh mang lại cơ chế suy
luận hoặc generalization khác biệt.

## 6. Quyết định theo cổng

| Cổng | Trạng thái | Lý do |
|---|---|---|
| Khác reconstruction về đại số | Chỉ đạt với head độc lập | Bản naive thất bại hoàn toàn. |
| Tác động lên đầu ra khi inference | Có thể đạt | Phải dùng decomposition trong residual cuối. |
| Target có nghĩa và nhận dạng được | Chưa đạt | Phụ thuộc generator; component target đến từ observation khác. |
| Dữ liệu replay đúng realization | Chưa đạt | CDD-11 hiện tại không có manifest. |
| Khoảng trống literature | Chưa đạt | Claim rộng va chạm trực tiếp với PRISM. |

### Hướng tiếp theo

Không huấn luyện interaction model hiện tại. Bước hợp lý tiếp theo là chọn một
claim hẹp mà PRISM chưa giải quyết, rồi mới thiết kế target và generator tương
ứng. Nếu vẫn giữ interaction, claim bắt buộc phải là một trong hai dạng có thể
bác bỏ rõ ràng:

1. một cơ chế deployed dùng **directed non-additive interaction** và tốt hơn
   các đối chứng multi-task/parameter-matched trên held-out compositions; hoặc
2. một kết quả phân tích cho thấy interaction target dự đoán được failure hay
   generalization tốt hơn primitive/mixture embeddings của PRISM.

Nếu không viết được công thức và protocol phân biệt rõ một trong hai claim này,
nên loại hướng interaction supervision thay vì mở pilot GPU.
