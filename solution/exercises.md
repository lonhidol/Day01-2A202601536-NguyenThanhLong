# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 14h00–18h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.7, 1.2 và 1.8 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Hà Nội."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi? Ở mức nào phản hồi bắt đầu
kém mạch lạc?** (2–3 câu)
> *Các phản hồi đều bắt đầu với cùng 1 câu "Một sự thật thú vị về Hà Nội là thành phố này có " và bắt đầu đưa các dẫn chứng khác nhau. Nhưng hiện tại thì bắt đầu ở Temperature 0.7 thì đã bắt đầu xảy ra vấn đề khi lại miêu tả cụ thể 1 di tích riêng chứ không phải 1 đặc điểm của Hà Nội*

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho trợ lý soạn thảo hợp đồng pháp lý,
và bao nhiêu cho trợ lý viết slogan quảng cáo? Giải thích khác biệt.**
> *Với các thủ tục pháp lý, mình sẽ cân nhắc sử dụng temperature ở mức 0 như vậy sẽ đảm bảo tính chính xác tuyệt đối, cứng nhắc và rành mạch. Và ngược lại đối với trợ lý viết slogan quảng cáo mình sẽ dùng temperature ở mức cao hơn điều này phù hợp với sự sáng tạo, linh hoạt và bay bổng. Sự khác biệt này là do mức độ temperature sẽ ảnh hưởng đến sự ngẫu nhiên, temperatura cao sẽ khiến ngữ nghĩa có sự ngẫu nhiên lớn và ngược lại.*

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 20.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 2 lần,
mỗi lần trung bình ~500 token đầu ra.

**Ước tính chi phí mỗi ngày của model lớn so với model nhỏ cho workload này
(dựa trên bảng giá trong template). Nêu một trường hợp model lớn xứng đáng
với chi phí và một trường hợp model nhỏ là lựa chọn đúng:**
> *Ước tính chi phí tổng số lượng token đầu ra sẽ là 20.000.000 tokens. Đối với model lớn như gpt-4o sẽ tiêu thụ 200$ 1 ngày và với model nhỏ gpt-4o-mini sẽ khoảng 12$ 1 ngày. Đối với model lớn, mình nghĩ sẽ phù hợp với những việc như là dịch thuật slide hoặc giáo trình yêu cầu phải dịch thuật và hiểu ngữ nghĩa 1 cách tường minh còn đối với module nhỏ sẽ phù hợp với những tác vụ thông thường nhưng hỏi những quán cafe xung quanh, phù hợp với bản thân.*

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích máy học (machine learning) là gì?"** nhưng hai system prompt
khác nhau:
- "Bạn là một nhà thơ, trả lời mọi thứ bằng hình ảnh ví von, tránh thuật ngữ."
- "Bạn là kỹ sư phần mềm senior, trả lời chính xác, có ví dụ code khi phù hợp."

**Hai phản hồi khác nhau như thế nào (giọng văn, độ dài, mức kỹ thuật)?
Từ đó rút ra system prompt điều khiển được những khía cạnh nào của phản hồi?**
(3–4 câu)
> *Đối với phản hồi đầu tiên Nhà thơ thì cho thấy giọng văn mang đậm tính thơ hơn với độ dài không nhiều và mức kỹ thuật chỉ là mơ hồ như là trên góc nhìn cá nhân, còn đối với Senior Engineer thì cho thấy 1 góc nhìn mang tính khái niệm hơn, rõ ràng, tường minh với độ dài lớn để có thể phù hợp với persona. Từ đấy có thể thấy system prompt điều khiển được những khía cạnh có thể điều khiển như là độ sâu, nhân cách, giọng điệu và định dạng đầu ra của văn bản.*

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~150 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Nếu dùng ước lượng thô để dự
toán ngân sách API cho ứng dụng tiếng Việt, bạn sẽ dự toán thiếu hay thừa —
và vì sao?**
> *Với 1 đoạn có số lượng từ thực tế là 166 từ, ước lượng thô cho thấy khoảng ~221 tokens và đếm được ~280 tokens, mức độ chênh lệch ở đây cho thấy khoảng 59 tokens. Đối với ứng dụng tiếng Việt token tiêu thụ sẽ cao hơn đối với tiếng Anh vậy như vậy nếu dùng ước lượng thô như trên để dự toán ngân sách API cho ứng dụng tiếng Việt, sẽ trở nên thiếu tokens trầm trọng*

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Xét ba ứng dụng: (a) chatbot văn bản, (b) trợ lý giọng nói đọc to phản hồi,
(c) pipeline dịch tài liệu chạy ngầm ban đêm. Ứng dụng nào hưởng lợi nhiều
nhất từ streaming, ứng dụng nào không cần — và tại sao?** (1 đoạn văn)
> *Câu trả lời của bạn*

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**Khi API quá tải và hàng nghìn client cùng retry, exponential backoff giúp
gì so với delay cố định? Tra cứu thêm: kỹ thuật "jitter" (thêm độ trễ ngẫu
nhiên) giải quyết vấn đề gì còn sót lại?**
> *Câu trả lời của bạn*

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Viết lại system prompt bạn dùng cho trợ lý của mình. Chỉ ra 2 chỗ trong
prompt mà nếu xóa đi, hành vi trợ lý sẽ thay đổi rõ rệt — và mô tả thay đổi
đó:**
> *Câu trả lời của bạn*

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn giữ history 4 lượt cuối. Hãy mô tả một tình huống hội thoại
cụ thể mà giới hạn này khiến trợ lý trả lời sai/mất ngữ cảnh, và đề xuất một
cách khắc phục (ví dụ: tóm tắt các lượt cũ, tăng giới hạn có chọn lọc...):**
> *Câu trả lời của bạn*

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên GitHub cá nhân và nộp link repo vào vlearn (theo hướng dẫn README)
