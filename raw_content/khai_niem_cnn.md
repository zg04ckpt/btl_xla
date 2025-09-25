1. Khái niệm
Mạng nơ-ron tích chập (Convolutional Neural Network – CNN, hay ConvNet) là một dạng đặc biệt của mạng nơ-ron nhân tạo được thiết kế để xử lý dữ liệu có cấu trúc dạng lưới, đặc biệt là hình ảnh. Thay vì kết nối đầy đủ giữa mọi nơ-ron, CNN sử dụng phép tích chập (convolution) để quét qua dữ liệu và trích xuất các đặc trưng quan trọng.
CNN được xem là nền tảng cho nhiều thành tựu lớn trong thị giác máy tính từ năm 2012 (khi AlexNet chiến thắng cuộc thi ImageNet).
2. Đặc trưng của CNN
Cấu trúc nhiều tầng (layered structure): CNN gồm nhiều tầng xử lý liên tiếp, từ trích chọn đặc trưng cơ bản (cạnh, góc, đường viền) đến đặc trưng phức tạp (hình dạng, đối tượng).
Chia sẻ trọng số (weight sharing): cùng một bộ lọc (filter/kernel) được áp dụng cho toàn bộ ảnh → giảm số lượng tham số cần học.
Khai thác tính cục bộ (local connectivity): mỗi nơ-ron chỉ kết nối với một vùng nhỏ (receptive field), giúp mô hình tập trung vào đặc trưng cục bộ.
Bất biến dịch chuyển (translation invariance): nhận dạng đối tượng ngay cả khi chúng thay đổi vị trí trong ảnh.

3. Các thành phần chính
Convolutional layer: dùng các bộ lọc để phát hiện đặc trưng cục bộ.
Pooling layer (ví dụ: max pooling): giảm kích thước dữ liệu, tăng tính bất biến đối với dịch chuyển và giảm chi phí tính toán.
Activation function (thường là ReLU): thêm phi tuyến tính để mô hình hóa các đặc trưng phức tạp.
Fully Connected layer: kết hợp các đặc trưng đã học để đưa ra dự đoán cuối cùng.
4. Điểm mạnh của CNN
Tự động trích chọn đặc trưng: không cần thiết kế thủ công như trong học máy truyền thống.
Độ chính xác cao: đặc biệt trong các bài toán nhận dạng ảnh và thị giác máy tính.
Khả năng mở rộng: áp dụng cho ảnh, video, dữ liệu chuỗi (speech, NLP).
Giảm số lượng tham số: nhờ cơ chế chia sẻ trọng số, CNN có thể huấn luyện hiệu quả trên dữ liệu lớn.
5. Ứng dụng của CNN
Nhận dạng chữ viết tay và chữ số (MNIST).
Phân loại ảnh, nhận dạng đối tượng (ImageNet).
Phân đoạn ảnh (image segmentation) trong y tế, xe tự hành.
Xử lý video và nhận dạng hành động.
Ứng dụng ngoài thị giác máy tính: phân tích tín hiệu âm thanh, chuỗi thời gian, văn bản.
