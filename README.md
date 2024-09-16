# Churn-Iran-Report
## Giới thiệu dữ liệu
Bộ dữ liệu này được thu thập ngẫu nhiên từ cơ sở dữ liệu của một công ty viễn thông Iran trong khoảng thời gian 12 tháng. Tổng cộng có 3150 dòng dữ liệu, mỗi dòng đại diện cho một khách hàng, chứa thông tin cho 13 cột. Các thuộc tính trong bộ dữ liệu này bao gồm số lần gọi thất bại, tần suất SMS, số lần khiếu nại, số cuộc gọi khác nhau, thời gian đăng ký, nhóm tuổi, số tiền cước, loại dịch vụ, số giây sử dụng, trạng thái, tần suất sử dụng và giá trị khách hàng.

Tất cả các thuộc tính trừ thuộc tính churn đều là dữ liệu tổng hợp của 9 tháng đầu tiên. Nhãn churn là trạng thái của khách hàng vào cuối 12 tháng. Ba tháng còn lại là khoảng thời gian lập kế hoạch.
## Tiền xử lý dữ liệu
### EDA - Phân tích Khám phá Dữ liệu

| Cột                       | Số Lượng Không Thiếu | Kiểu Dữ Liệu |
|---------------------------|-----------------------|--------------|
| Call Failure              | 3150                  | int64        |
| Complains                 | 3150                  | int64        |
| Subscription Length       | 3150                  | int64        |
| Charge Amount             | 3150                  | int64        |
| Seconds of Use            | 3150                  | int64        |
| Frequency of use          | 3150                  | int64        |
| Frequency of SMS          | 3150                  | int64        |
| Distinct Called Numbers   | 3150                  | int64        |
| Age Group                 | 3150                  | int64        |
| Tariff Plan               | 3150                  | int64        |
| Status                    | 3150                  | int64        |
| Age                       | 3150                  | int64        |
| Customer Value            | 3150                  | float64      |
| Churn                     | 3150                  | int64        |

#### Bảng Mô Tả Cột Dữ Liệu

| Cột                      | Mô Tả                                                                                         | Kiểu Dữ Liệu |
|--------------------------|-----------------------------------------------------------------------------------------------|--------------|
| Call Failure             | Số lần gọi thất bại của từng khách hàng.                                                     | int64        |
| Complains                | Số lần khiếu nại được gửi từ từng khách hàng.                                                | int64        |
| Subscription Length      | Thời gian sử dụng dịch vụ (tháng) của từng khách hàng.                                        | int64        |
| Charge Amount            | Số tiền đã tính cho từng khách hàng.                                                           | int64        |
| Seconds of Use           | Tổng số giây sử dụng dịch vụ của từng khách hàng.                                             | int64        |
| Frequency of Use         | Tần suất sử dụng dịch vụ của từng khách hàng.                                                  | int64        |
| Frequency of SMS         | Tần suất sử dụng tin nhắn SMS của từng khách hàng.                                             | int64        |
| Distinct Called Numbers  | Số lượng số điện thoại khác nhau mà từng khách hàng đã gọi.                                   | int64        |
| Age Group                | Nhóm tuổi của từng khách hàng.                                                                 | int64        |
| Tariff Plan              | Kế hoạch giá cước mà từng khách hàng đang sử dụng.                                              | int64        |
| Status                   | Trạng thái hiện tại của từng khách hàng.                                                       | int64        |
| Age                      | Tuổi của từng khách hàng.                                                                      | int64        |
| Customer Value           | Giá trị khách hàng được tính toán dựa trên các chỉ số khác.                                    | float64      |
| Churn                    | Chỉ số churn (tỷ lệ khách hàng chuyển đổi) của từng khách hàng.                              | int64        |

- **Số cột**: 14
- **Số cột kiểu `int64`**: 13
- **Số cột kiểu `float64`**: 1
- **Số hàng**: 3150
- **Bộ dữ liệu không chứa giá trị NaN.**
Tập dữ liệu này cung cấp thông tin chi tiết về hành vi của khách hàng và giá trị của họ, giúp phân tích và dự đoán các xu hướng liên quan đến việc sử dụng dịch vụ và tỷ lệ churn.
### Train Test Split - Chia tập dữ liệu
X là cà các đặc trưng đầu vào của mô hình.

y là biến mục tiêu (target variable).

Tập dữ liệu được chia thành 2 phần:  20% của dữ liệu sẽ được dành cho tập kiểm tra và 80% còn lại sẽ được dùng cho huấn luyện

### Scaling - Chuẩn hóa dữ liệu

### Resampling - Điều chỉnh tỷ lệ mẫu
Số lượng các mẫu trong dữ liệu gốc trước khi resampling: 2655 mẫu thuộc lớp 0 và 495 mẫu thuộc lớp 1.

Số lượng các mẫu sau khi thực hiện resampling:  cả hai lớp đều có 385 mẫu.
## Huấn luyện mô hình
### Mô hình Logistic Regression

### Mô hình Decision Tree

### Mô hình Random Forest

### Mô hình K-Neighbor KNN

### Mô hình Support Vector Classifier - SVC 
### Chọn mô hình phù hợp

#### Kết Quả Đánh Giá Mô Hình

Dưới đây là bảng đánh giá hiệu suất của các mô hình khác nhau:

| Model                     | Accuracy | F1 Score | Precision | Recall |
|---------------------------|----------|----------|-----------|--------|
| Logistic Regression       | 0.8159   | 0.6234   | 0.4848    | 0.8727 |
| Decision Tree             | 0.8159   | 0.6234   | 0.4848    | 0.8727 |
| Random Forest             | 0.8825   | 0.7338   | 0.6071    | 0.9273 |
| K-Neighbors               | 0.9016   | 0.7597   | 0.6622    | 0.8909 |
| Support Vector Classifier | 0.8968   | 0.7601   | 0.6398    | 0.9364 |

#### Phân Tích Kết Quả

Dựa trên bảng đánh giá, mô hình có hiệu suất tốt nhất là **Support Vector Classifier (SVC)** với các chỉ số như sau:

- **Accuracy**: 0.8968
- **F1 Score**: 0.7601
- **Precision**: 0.6398
- **Recall**: 0.9364

Mô hình này có tỷ lệ dự đoán chính xác (Accuracy) cao và độ F1 Score tương đối cao, đồng thời cân bằng khá tốt giữa Precision và Recall. Điều này cho thấy mô hình có khả năng dự đoán tốt trên cả các trường hợp dương tính và âm tính trong dữ liệu của bạn.


