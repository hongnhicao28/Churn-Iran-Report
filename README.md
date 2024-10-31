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
#### Tìm siêu tham số
C (Inverse of regularization strength): Tham số điều chỉnh mức độ phạt của hàm mất mát trong quá trình huấn luyện.  
Penalty (L1, L2): Loại hàm regularization được áp dụng (L1 - Lasso, L2 - Ridge).  
Solver: Thuật toán sử dụng để tối ưu hóa.  

```
Define the parameter grid
param_grid_LR = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga', 'lbfgs'],
    'penalty': ['l2']
}
```
#### Huấn luyện mô hình
```
model_LR = LogisticRegression()
```
#### Đánh giá mô hình

**Kết quả**
**accuracy_LR:* 0.8158730158730159  
**f1_LR:* 0.6233766233766235  
**precision_LR:* 0.48484848484848486  
**recall_LR:* 0.872727272727272  
Ma trận nhầm lẫn
- Có 418 điểm dữ liệu thuộc lớp negative (lớp 0) và được dự đoán đúng là negative (TN).
- Có 102 điểm dữ liệu thuộc lớp negative nhưng bị dự đoán là positive (FP).
- Có 14 điểm dữ liệu thuộc lớp positive nhưng bị dự đoán là negative (FN).
- Có 96 điểm dữ liệu thuộc lớp positive và được dự đoán đúng là positive (TP).

### Mô hình Decision Tree

### Mô hình Random Forest

### Mô hình K-Neighbor KNN

### Mô hình Support Vector Classifier - SVC 
```
accuracy_SVC = accuracy_score(y_test, y_pred_SVC)
f1_SVC = f1_score(y_test, y_pred_SVC)
precision_SVC = precision_score(y_test, y_pred_SVC)
recall_SVC = recall_score(y_test, y_pred_SVC)
print("accuracy_SVC:", accuracy_SVC)
print("f1_SVC:", f1_SVC)
print("precision_SVC:", precision_SVC)
print("recall_SVC:", recall_SVC)
```
**Kết quả**
*accuracy_SVC:* 0.8968253968253969  
*f1_SVC:* 0.7601476014760147  
*precision_SVC:* 0.639751552795031  
*recall_SVC:* 0.9363636363636364  
Ma trận nhầm lẫn
- Có 462 điểm dữ liệu thuộc lớp negative (lớp 0) và được dự đoán đúng là negative (TN).
- Có 58 điểm dữ liệu thuộc lớp negative nhưng bị dự đoán là positive (FP).
- Có 7 điểm dữ liệu thuộc lớp positive nhưng bị dự đoán là negative (FN).
- Có 103 điểm dữ liệu thuộc lớp positive và được dự đoán đúng là positive (TP).

### Chọn mô hình phù hợp
#### Kết Quả Đánh Giá Mô Hình
Dưới đây là bảng đánh giá hiệu suất của các mô hình khác nhau:
##### Classification Reports for Machine Learning Models
Dưới đây là các báo cáo phân loại (Classification Reports) cho các mô hình khác nhau được đánh giá.  

---

###### Logistic Regression
| Metric         | Class 0 | Class 1 | Accuracy | Macro Avg | Weighted Avg |
|----------------|---------|---------|----------|-----------|--------------|
| Precision      | 0.97    | 0.48    | 0.82     | 0.73      | 0.88         |
| Recall         | 0.80    | 0.87    |          | 0.84      | 0.82         |
| F1-Score       | 0.88    | 0.62    |          | 0.75      | 0.83         |
| Support        | 520     | 110     |          | 630       | 630          |

---

###### Decision Tree
| Metric         | Class 0 | Class 1 | Accuracy | Macro Avg | Weighted Avg |
|----------------|---------|---------|----------|-----------|--------------|
| Precision      | 0.97    | 0.48    | 0.82     | 0.73      | 0.88         |
| Recall         | 0.80    | 0.87    |          | 0.84      | 0.82         |
| F1-Score       | 0.88    | 0.62    |          | 0.75      | 0.83         |
| Support        | 520     | 110     |          | 630       | 630          |

---

###### Random Forest
| Metric         | Class 0 | Class 1 | Accuracy | Macro Avg | Weighted Avg |
|----------------|---------|---------|----------|-----------|--------------|
| Precision      | 0.98    | 0.60    | 0.88     | 0.79      | 0.91         |
| Recall         | 0.87    | 0.91    |          | 0.89      | 0.88         |
| F1-Score       | 0.92    | 0.72    |          | 0.82      | 0.89         |
| Support        | 520     | 110     |          | 630       | 630          |

---

###### K-Neighbors
| Metric         | Class 0 | Class 1 | Accuracy | Macro Avg | Weighted Avg |
|----------------|---------|---------|----------|-----------|--------------|
| Precision      | 0.98    | 0.66    | 0.90     | 0.82      | 0.92         |
| Recall         | 0.90    | 0.89    |          | 0.90      | 0.90         |
| F1-Score       | 0.94    | 0.76    |          | 0.85      | 0.91         |
| Support        | 520     | 110     |          | 630       | 630          |

---

###### Support Vector Classifier
| Metric         | Class 0 | Class 1 | Accuracy | Macro Avg | Weighted Avg |
|----------------|---------|---------|----------|-----------|--------------|
| Precision      | 0.99    | 0.64    | 0.90     | 0.81      | 0.92         |
| Recall         | 0.89    | 0.94    |          | 0.91      | 0.90         |
| F1-Score       | 0.93    | 0.76    |          | 0.85      | 0.90         |
| Support        | 520     | 110     |          | 630       | 630          |

---

> **Ghi chú**: "Class 0" và "Class 1" tương ứng với các nhãn phân loại trong tập dữ liệu, với độ chính xác (accuracy), độ bao phủ (recall), và F1-Score được cung cấp cho từng mô hình.

#### Phân Tích Kết Quả
Dựa trên bảng đánh giá, mô hình có hiệu suất tốt nhất là **Support Vector Classifier (SVC)** với các chỉ số như sau:
- **Accuracy**: 0.8968
- **F1 Score**: 0.7601
- **Precision**: 0.6398
- **Recall**: 0.9364

Mô hình này có tỷ lệ dự đoán chính xác (Accuracy) cao và độ F1 Score tương đối cao, đồng thời cân bằng khá tốt giữa Precision và Recall. Điều này cho thấy mô hình có khả năng dự đoán tốt trên cả các trường hợp dương tính và âm tính trong dữ liệu.


