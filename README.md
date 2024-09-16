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


