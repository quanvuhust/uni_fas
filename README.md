# uni_fas
5th Chalearn Face Anti-spoofing Workshop and Challenge@CVPR2024

# Dataset structure
```
datasets/
    ├── phase1
    │   ├── p1
    │   ├── p2.1
    │   ├── p2.2
    ├── phase2
    │   ├── p1
    │   ├── p2.1
    │   ├── p2.2
```
# Weight structure
```
weights/
    ├── exp_5
    │   ├── p1
    │   ├── p2.1
    │   ├── p2.2
```
# Install env
```
chmod +x install_env.sh
./install_env.sh
```
# Prepare dataset
```
python create_csv_dev.py
python create_csv_test.py
python create_csv_train.py
```
# Training
```
python code/train.py --exp exp_5
```
# Create submission
```
python code/predict.py
```
# Authors
Vũ Minh Quân (quanvm4@viettel.com.vn)\
Hoàng Văn Cường (cuonghv28@viettel.com.vn)\
Nguyễn Thị Ánh (anhnt21@viettel.com.vn)