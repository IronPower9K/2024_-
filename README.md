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

<Default>
<img width="419" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/e95716cb-d9b3-4ad2-9da7-6daff9cf7af0">



3. FLANN.py
<결과>

<img width="1065" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/7943a152-38d2-4a10-b683-b95e3b827f87">

<Terminal>

<img width="1067" alt="image" src="https://github.com/IronPower9K/2024_computer_vision/assets/114505607/5cbeeeea-ee2c-4375-a83c-07c5cc3b4159">




4. Homography.py

-> FLANN을 이용하여 Matching후 Homography를 사용하여 Matching된 영역 표시 및 아웃라이어 검출

![Uploading image.png…]()




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
  
     
