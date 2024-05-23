train 5 (2024.05.22)

1.py

MNIST 데이터셋 로드 및 전처리:
MNIST 데이터셋을 로드
데이터셋의 차원을 CNN 입력 형태로 변환(28x28 이미지, 1채널).
픽셀 값을 [0, 1] 범위로 정규화
레이블을 원-핫 인코딩 형태로 변환

CNN 모델 구축:
CNN 모델을 Sequential 방식으로 구축
3개의 Conv2D 레이어와 MaxPooling2D 레이어를 추가
Flatten 레이어를 추가하여 다차원 출력을 1차원으로 변환
2개의 Dense 레이어를 추가하여 출력 노드를 정의. 마지막 레이어는 10개의 클래스에 대한 softmax 활성화 함수를 사용.

모델 컴파일 및 학습:
모델을 컴파일. 손실 함수는 'categorical_crossentropy', 옵티마이저는 'Adam', 평가는 'accuracy'를 사용.
학습 데이터를 사용하여 모델을 학습. 배치 크기는 128, 총 에포크 수는 30, 검증 데이터는 테스트 데이터로 설정.

![1](https://github.com/IronPower9K/2024_computer_vision/assets/114505607/0e518e2f-ae54-4213-be00-c11e1de6134e)

2.py

CNN 모델 구축:
CNN 모델을 Sequential 방식으로 구축
여러 개의 Conv2D, MaxPooling2D, Dropout 레이어를 추가
Flatten 레이어를 추가하여 다차원 출력을 1차원으로 변환
2개의 Dense 레이어를 추가하여 출력 노드를 정의 마지막 레이어는 10개의 클래스에 대한 softmax 활성화 함수를 사용

![2](https://github.com/IronPower9K/2024_computer_vision/assets/114505607/3f48af74-3fef-483d-b2cf-f7991ac8a156)

![2_loss](https://github.com/IronPower9K/2024_computer_vision/assets/114505607/dc76a4c0-4a5b-41e5-baf7-6aa2271cf907)

3.py



![3](https://github.com/IronPower9K/2024_computer_vision/assets/114505607/215337d3-626b-485a-83f0-a30925497d65)

4.py

데이터셋의 픽셀 값을 [0, 1] 범위로 정규화
훈련 데이터셋에서 첫 15개의 이미지를 시각화, 각 이미지의 실제 클래스 이름을 제목으로 표시
생성된 이미지와 레이블을 시각화
각 반복마다 배치 크기만큼의 이미지를 생성하고 시각화
3번 반복하여 다양한 증강 결과를 시각화

![4](https://github.com/IronPower9K/2024_computer_vision/assets/114505607/e5e2602a-b792-4d81-9ee6-4ab5a0f82d3c)

![4-2](https://github.com/IronPower9K/2024_computer_vision/assets/114505607/b5f397d2-2078-4b27-ae87-d40f42ecfbe9)

![4-3](https://github.com/IronPower9K/2024_computer_vision/assets/114505607/1c5ae768-6554-4da6-9b53-af1a076dd8b6)

![4-4](https://github.com/IronPower9K/2024_computer_vision/assets/114505607/2f3e391c-c2e7-4e89-92d5-7b83f879cff4)




5.py

모델 로드: 사전 학습된 ResNet50 모델을 로드하여 이미지넷 데이터셋의 가중치를 사용
이미지 로드 및 전처리: 'rabbit.jpg' 이미지를 로드 ResNet50 모델의 입력 형식에 맞게 전처리
예측 수행: 모델을 사용하여 이미지를 예측, 상위 5개의 예측 결과를 디코딩하여 출력
이미지에 결과 표시: 예측 결과를 이미지에 텍스트로 표시, 화면에 시각화

![5](https://github.com/IronPower9K/2024_computer_vision/assets/114505607/7815172c-3fd1-4bd5-95a4-74a4e2521012)

![5_log](https://github.com/IronPower9K/2024_computer_vision/assets/114505607/7febc4c0-1f98-4de1-bf86-98b10533acdf)

6.py

사전 학습된 DenseNet121 모델을 백본으로 사용하고, 최상위 레이어를 제거
Rescaling 레이어를 추가하여 입력 이미지를 정규화
DenseNet121의 출력 텐서를 1차원으로 평탄화, 추가적인 Dense 레이어와 Dropout 레이어를 추가
마지막 레이어는 120개의 클래스에 대해 softmax 활성화 함수를 사용

![6_acc](https://github.com/IronPower9K/2024_computer_vision/assets/114505607/cb7668c2-eb07-460e-a98d-6c4c26910963)

![6_loss](https://github.com/IronPower9K/2024_computer_vision/assets/114505607/8459a96c-f743-4e42-aa9c-3044827a10b5)

7.py

(source picture)

![1685251710889870](https://github.com/IronPower9K/2024_computer_vision/assets/114505607/38ddd862-de24-4c4b-b00b-0a91583c6f8c)

![7](https://github.com/IronPower9K/2024_computer_vision/assets/114505607/3f6fa635-d75a-4557-9ea6-ffca71f197d6)




실습 4 (2024.05.07)
1.py

MNIST 데이터셋 로드 및 시각화:

ds.mnist.load_data()를 사용하여 MNIST 데이터셋을 로드
로드된 데이터셋의 크기를 출력
첫 10개의 이미지를 시각화하고, 각 이미지의 실제 숫자 레이블을 제목으로 표시
CIFAR-10 데이터셋 로드 및 시각화:

ds.cifar10.load_data()를 사용하여 CIFAR-10 데이터셋을 로드
로드된 데이터셋의 크기를 출력
CIFAR-10 클래스 이름을 정의
첫 10개의 이미지를 시각화하고, 각 이미지의 실제 클래스 이름을 제목으로 표시

![1_pic](https://github.com/IronPower9K/2024_computer_vision/assets/114505607/dd333331-a50f-4d21-ad16-76eb530a9a50)

![1_pic_2](https://github.com/IronPower9K/2024_computer_vision/assets/114505607/278f1f16-e827-471f-8faa-4473a499f417)

2.py

데이터 로드 및 전처리:

ds.mnist.load_data()로 MNIST 데이터셋을 로드
이미지를 28x28에서 784차원 벡터로 변환
이미지 데이터를 0에서 1 사이의 값으로 정규화
레이블을 원-핫 인코딩 형식으로 변환
모델 구성 및 컴파일:

Sequential 모델을 사용합니다.
첫 번째 층은 512개의 뉴런과 tanh 활성화 함수를 가진 밀집 층
두 번째 층은 10개의 뉴런과 softmax 활성화 함수를 가진 출력 층
모델을 MSE 손실 함수와 SGD 옵티마이저로 컴파일하고, 평가 메트릭으로 accuracy를 설정
모델 훈련:

모델을 훈련 데이터로 50 에포크 동안 훈련
배치 크기는 128로 설정하고, 검증 데이터로 테스트 셋을 사용
모델 평가 및 결과 출력:

테스트 데이터를 사용하여 모델을 평가
평가 결과인 정확도를 출력

![실습2](https://github.com/IronPower9K/2024_computer_vision/assets/114505607/a8123887-1f22-414b-adad-a852eb9b6927)

3.py

데이터 로드 및 전처리:

ds.mnist.load_data()로 MNIST 데이터셋을 로드
이미지를 28x28에서 784차원 벡터로 변환
이미지 데이터를 0에서 1 사이의 값으로 정규화
레이블을 원-핫 인코딩 형식으로 변환
모델 구성 및 컴파일:

Sequential 모델을 사용합니다.
첫 번째 층은 512개의 뉴런과 tanh 활성화 함수를 가진 밀집 층
두 번째 층은 10개의 뉴런과 softmax 활성화 함수를 가진 출력 층
모델을 MSE 손실 함수와 Adam 옵티마이저(학습률 0.001)로 컴파일하고, 평가 메트릭으로 accuracy를 설정
모델 훈련:

모델을 훈련 데이터로 50 에포크 동안 훈련
배치 크기는 128로 설정하고, 검증 데이터로 테스트 셋을 사용
모델 평가 및 결과 출력:

테스트 데이터를 사용하여 모델을 평가
평가 결과인 정확도를 출력

![실습3](https://github.com/IronPower9K/2024_computer_vision/assets/114505607/660b0e12-fc0d-45f0-8cbd-e7b5b5618fd1)

4.py


데이터 로드 및 전처리:

MNIST 데이터셋을 ds.mnist.load_data()로 로드
이미지를 28x28에서 784차원 벡터로 변환
이미지 데이터를 0에서 1 사이의 값으로 정규화
레이블을 원-핫 인코딩 형식으로 변환
SGD 옵티마이저를 사용한 모델:

Sequential 모델을 구성하고, 첫 번째 층으로 512개의 뉴런과 tanh 활성화 함수를 가진 밀집(Dense) 층을 추가
출력 층으로 10개의 뉴런과 softmax 활성화 함수를 가진 층을 추가
모델을 MSE 손실 함수와 SGD 옵티마이저(학습률 0.01)로 컴파일하고, 평가 메트릭으로 accuracy를 설정
모델을 50 에포크 동안 훈련하고, 훈련 및 검증 정확도를 저장
테스트 데이터로 모델을 평가하고, 최종 정확도를 출력
Adam 옵티마이저를 사용한 모델:

위와 동일한 모델 구조를 사용하되, 옵티마이저로 Adam(학습률 0.001)을 사용
모델을 50 에포크 동안 훈련하고, 훈련 및 검증 정확도를 저장
테스트 데이터로 모델을 평가하고, 최종 정확도를 출력

![실습4_sgd](https://github.com/IronPower9K/2024_computer_vision/assets/114505607/7b8d439e-ee73-4a87-8e0c-c944cecaad25)

![실습4](https://github.com/IronPower9K/2024_computer_vision/assets/114505607/3b711aa7-76eb-47cf-9c53-11459348d63a)

5.py

데이터 로드 및 전처리:

MNIST 데이터셋을 ds.mnist.load_data()로 로드
이미지를 28x28에서 784차원 벡터로 변환
이미지 데이터를 0에서 1 사이의 값으로 정규화
레이블을 원-핫 인코딩 형식으로 변환
심층 신경망 모델 구성:

Sequential 모델을 사용합니다.
첫 번째 층은 1024개의 뉴런과 relu 활성화 함수를 가진 밀집(Dense) 층
두 번째와 세 번째 층은 각각 512개의 뉴런과 relu 활성화 함수를 가진 밀집 층
출력 층은 10개의 뉴런과 softmax 활성화 함수를 가진 층
모델 컴파일:

모델을 categorical_crossentropy 손실 함수와 Adam 옵티마이저(학습률 0.0001)로 컴파일
평가 메트릭으로 accuracy를 설정
모델 훈련:

모델을 훈련 데이터(x_train, y_train)로 50 에포크 동안 훈련
배치 크기는 128로 설정하고, 검증 데이터(x_test, y_test)를 사용
훈련 과정의 정확도와 손실을 저장
모델 평가:

테스트 데이터를 사용하여 모델을 평가하고, 최종 정확도를 출력
모델 저장:

훈련된 모델을 dmlp_trained.h5 파일로 저장
정확도 및 손실 그래프 시각화:

훈련과 검증 과정에서의 정확도 변화를 그래프로 시각화
훈련과 검증 과정에서의 손실 변화를 그래프로 시각화

![Unknown-5](https://github.com/IronPower9K/2024_computer_vision/assets/114505607/73631987-2baa-47c9-aa97-36aa4e51454c)

![Unknown-6](https://github.com/IronPower9K/2024_computer_vision/assets/114505607/165ef8a6-c1dd-457a-be1a-c0a59805f88c)

6.py

데이터 로드 및 전처리:

CIFAR-10 데이터셋을 ds.cifar10.load_data()로 로드
이미지 데이터를 32x32x3에서 3072차원 벡터로 변환
이미지 데이터를 0에서 1 사이의 값으로 정규화
레이블을 원-핫 인코딩 형식으로 변환
심층 신경망 모델 구성:

Sequential 모델을 사용합니다.
첫 번째 층은 1024개의 뉴런과 relu 활성화 함수를 가진 밀집(Dense) 층
두 번째와 세 번째 층은 각각 512개의 뉴런과 relu 활성화 함수를 가진 밀집 층
출력 층은 10개의 뉴런과 softmax 활성화 함수를 가진 층
모델 컴파일:

모델을 categorical_crossentropy 손실 함수와 Adam 옵티마이저(학습률 0.0001)로 컴파일
평가 메트릭으로 accuracy를 설정
모델 훈련:

모델을 훈련 데이터(x_train, y_train)로 50 에포크 동안 훈련
배치 크기는 128로 설정하고, 검증 데이터(x_test, y_test)를 사용
훈련 과정의 정확도와 손실을 저장
모델 평가:

테스트 데이터를 사용하여 모델을 평가하고, 최종 정확도를 출력
정확도 및 손실 그래프 시각화:

훈련과 검증 과정에서의 정확도 변화를 그래프로 시각화
훈련과 검증 과정에서의 손실 변화를 그래프로 시각화


<default>

![Accurecy_6](https://github.com/IronPower9K/2024_computer_vision/assets/114505607/9a97ed13-22fd-4923-b251-4c6fa3951466)

![loss_6](https://github.com/IronPower9K/2024_computer_vision/assets/114505607/fbcb022f-b76d-4c31-9129-aa182aae6028)

6_modify.py


성능 개선 방법
-> 각  Layer당 정규화 적용
-> Dropout 적용 시 은닉층 별로 차등하여 적용
-> Overfitting을 해결 하기 위해 첫번째 은닉층보다 마지막 은닉층으로 갈수록 순차적으로 낮은 값을 제시함으로써 최대한 최초 은닉층에서 Dropout이 많이 되게한 후 서서히 Dropout양을 줄임으로써, 최대한 Overfitting을 방지하며, 출력층으로 전달되는 노드 값을 최대로 하기 위함.

=> Train Accuracy 와  Val Accuracy간 격차가 개선전에 비해 현저히 떨어짐을 확인함.

![Accurecy_6_add](https://github.com/IronPower9K/2024_computer_vision/assets/114505607/04931d6d-eda4-494c-8e4b-566bb3f9dfa7)


![loss_6_add](https://github.com/IronPower9K/2024_computer_vision/assets/114505607/fa791a26-42e0-4a0d-9f89-ae672a99a7e0)


7.py

모델 로드: 미리 훈련된 심층 신경망 모델(dmlp_trained.h5)을 로드

인터페이스 초기화 함수(reset): 인터페이스 이미지를 초기화하고, 5개의 숫자를 입력할 수 있는 상자를 생성

숫자 캡처 함수(grab_numerals): 인터페이스에서 숫자를 캡처하고, 각 숫자를 28x28 픽셀 크기로 조정하고 반전

숫자 출력 함수(show): 캡처한 숫자를 시각화

숫자 인식 함수(recognition): 캡처한 숫자를 모델에 입력하여 숫자를 인식하고, 이미지에 해당 숫자를 표시

마우스 이벤트 처리 함수(writing): 마우스 이벤트를 처리하여 사용자가 숫자를 입력

인터페이스 반복 루프: 사용자가 'e'를 누르면 초기화, 's'를 누르면 캡처한 숫자 표시, 'r'을 누르면 숫자 인식을 실행하고, 'q'를 누르면 종료

![image](https://github.com/IronPower9K/2024_computer_vision/assets/114505607/c8c5f9c7-267f-4903-b745-5d14f113ee40)
![image](https://github.com/IronPower9K/2024_computer_vision/assets/114505607/cba9a35e-4641-4245-8966-dff2423f8293)
![image](https://github.com/IronPower9K/2024_computer_vision/assets/114505607/8f82d1ca-41f6-4b5e-90ed-0d26a92b3114)









실습 3 (2024.04.30)
1. heris.py


<결과>
<img width="198" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/5d31f87e-cf03-403d-b497-bc3e57e61f81">

<Y방향 기울기 검출>

<img width="310" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/3b8394ef-f1ff-4cfe-8f76-1c1d9028766b">

<X방향 기울기 검출>

<img width="266" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/214e843f-1971-4455-bb14-f1a989785033">
<img width="382" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/c3efdaa7-61c9-4e90-9c91-6d6cb6ee25a4">
<img width="448" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/99febfa1-1d5c-4b4a-b0be-8c26fdaa29c9">

<Corner 특징 검출>

<img width="232" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/25455b23-7e74-42cd-9ea2-53b913cc6c1d">





2. SIFT.py



<nfeatures=0,nOctaveLayers=5>

<img width="954" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/3ad3d0b5-e579-41b8-8b3c-ab5bdfd671e3">

<nfeatures=0,nOctaveLayers=5,edgeThreshold=3,sigma=1.6>

=> edgeThreshold를 급격하게 줄임 (10 -> 3) 
=> 검출되는 FeaturePoint가 현저하게 줄어듦
<img width="954" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/eca5dd36-e326-4132-a785-d90706eb562c">

<nfeatures=0,nOctaveLayers=15,edgeThreshold=10,sigma=1.6>
=> nOctaveLayers를 급격하게 높임 (3 -> 15)
=> Layer가 많아짐에 따라 검출되는 Point가 많아짐
=> Layer가 깊어짐에 따라 정확도가 떨어지며, OverFitting우려가 됨
=> 복잡도와 여러 파라미터를 기반으로 그에 비례한 OctaveLayer 설정 필요

<img width="954" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/74446aca-5261-47fa-b183-4f399af0357c">


<nfeatures=1000,nOctaveLayers=5,edgeThreshold=10,sigma=1.6>
=> FeaturePoint 갯수 제한
=> 제한을 두지 않았을 결우에 비해 갯수감소 및 FeaturePoint가 검출될 때 Distance가 적은 FeaturePoint위주로 검출 확인

<img width="954" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/da40daf0-a970-4941-a4f4-a421b9a138f8">

<nfeatures=10000,nOctaveLayers=5,edgeThreshold=10,sigma=1.6>

=> nfeatures를 변경할 경우 특징점 검출 갯수에 영향을 끼침
<img width="956" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/7864177e-af31-4b35-8b0c-5d01a71abdec">



3. FLANN.py
<결과>

<img width="1065" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/7943a152-38d2-4a10-b683-b95e3b827f87">

<Terminal>

<img width="1067" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/5cbeeeea-ee2c-4375-a83c-07c5cc3b4159">




4. Homography.py

-> FLANN을 이용하여 Matching후 Homography를 사용하여 Matching된 영역 표시 및 아웃라이어 검출

<img width="1071" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/f6f8a2bf-348d-4d3d-acb2-eb241aad9962">





Opencv 실습 2 (2024.04.09)

  1. 1_sobel.py
    (Original)
    <img width="713" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/7c93e8be-dce6-447d-9289-3ea0d407826e">
    
    
  (Sobelx)

    
  <img width="713" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/7b295994-a47c-4856-b75e-9b1a7bf14e47">
    
  (Sobely)
    
   <img width="713" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/79921c95-8f41-43d4-bee4-d16391658bbc">
    
  (edge strength)
    
  <img width="713" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/6556ba83-3da4-4ad1-877a-f0ef66630658">


  2.2_canny.py
    (Original)
    
  <img width="713" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/37736efb-2fd8-455c-a4cc-409736545f58">
  
  (Canny1)
    
  <img width="713" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/7887d1e6-e8e7-40df-9a59-7e6a22b032a1">
  
  (Canny2)
    
   <img width="713" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/e55991d7-449b-4d84-9b63-ac5defb8b0d5">

  3.3_line_apple.py
  (apple detection)
    
  <img width="621" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/e9c8aca1-2900-4ab9-ae19-f191ab4813bc">

  3.3_line_soccer.py
  <img width="1429" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/b22027f3-6a59-4b87-8821-9a3814ea898d">


4.4_section_slic.py
  (coffee)
  
  <img width="598" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/afb4ac43-c8cf-4dc7-9cd1-0fba85913c15">
  
  (slic compact 20)
  
  <img width="598" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/02a8ff45-a61a-4021-b27c-9fb96cab0f6c">
  
  (slic compact 80)
  
  <img width="598" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/02a255c5-7cfd-40ce-86bd-b40c2bbc02d2">

4.4_optimization.py
(SLIC)

<img width="559" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/129ab86a-ef9a-4c49-8ec8-49a357d55d02">
<img width="598" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/2fe9de4e-e47b-4ef2-b9d9-2eda21c4f20b">

4-1.4_optimization_seperate.py

![image](https://github.com/IronPower9K/2024_computer_vision/assets/114505607/f7345689-514f-44f0-90c3-c44bf90df19c)

![image](https://github.com/IronPower9K/2024_computer_vision/assets/114505607/6458cc56-2be1-42d1-b00b-7135a7de88a0)


5_time_sharing_division.py

(select SOI)
<img width="601" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/4bdf2540-745c-454b-a7c1-56898d703f6a">

(dst)
<img width="601" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/2330818d-0e8d-4b0d-8201-cd9da697ee1f">

6_Area_characteristics.py

<img width="868" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/33b00926-c58c-4598-9f31-66563805d6c4">


(horse)

<img width="397" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/372d8966-aeef-45cc-8e47-ad641a21ede5">

(horse with contour)

<img width="397" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/1423d14d-3aeb-4ce0-93e8-e20d4e035bee">

(horse with line)

<img width="393" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/b63012f9-a142-4966-a8ef-557880da5bd0">


    




Opencv 실습 1 (2024.03.28)

  1. 2024_03_28_영상읽기_3.py 결과
     <img width="714" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/a3e59f2d-1e9d-40f5-bcdf-7183d7a3fbda">
     <img width="698" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/662c297e-485d-48ce-b5f1-fcb1a3cb9a64">
  
  
  2. 2024_03_28_영상읽고_크기축소_4.py 결과
     <img width="1431" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/d8da550d-e27e-41c9-b78e-c07a18d6df85">
  
     
  
  3. 2024_03_28_영상글씨_7_1.py 결과
     <img width="714" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/17c410ae-8b94-4dc2-9260-7b3ddc1ae436">
  
  
  4. 2024_03_28_영상글씨_7_2.py 결과
     <img width="714" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/560a4927-2716-4a0a-bcb2-364c0bf34a86">
  
  5. 2024_03_28_8.py 결과
     <img width="714" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/29e21705-c817-4c4d-8c93-abf17b3435d7">
  
  
  6. 2024_03_28_9.py 결과
     <img width="714" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/9eff2bb6-070d-4701-9337-44d02b5186e6">
  
     
