# 최적화 분할
import skimage
import numpy as np
import cv2
import time

img = skimage.data.coffee()     # 커피 이미지를 불러옴

start = time.time()     # 시간 측정 시작

# SLIC(Simple Linear Iterative Clustering) 알고리즘을 사용하여 이미지를 분할
# compactness는 영역 내부의 컨트롤을 결정하는 매개변수
# 값이 작을수록 영역이 크고, 값이 클수록 영역이 작아짐
# n_segments는 분할되는 영역의 개수를 나타냄
# start_label은 시작 레이블을 지정
slic = skimage.segmentation.slic(img, compactness=20, n_segments=600, start_label=1)

# RAG(Region Adjacency Graph)를 생성합니다.
g = skimage.graph.rag_mean_color(img, slic, mode="similarity")

# 정규화된 최소 컷을 계산하여 분할된 이미지를 얻습니다.
ncut = skimage.graph.cut_normalized(slic, g)

# 이미지를 분할하는데 걸리는 시간을 출력
print(img.shape, "image를 분할하는 데", time.time()-start, "초 소요")

# 분할된 이미지 주위에 경계를 표시
marking = skimage.segmentation.mark_boundaries(img, ncut)

# 0~255 사이의 정수로 변환
ncut_coffee = np.uint8(marking*255.0)

# 경계가 표시된 이미지를 표시
cv2.imshow("Normalized cut", cv2.cvtColor(ncut_coffee, cv2.COLOR_BGR2RGB))

cv2.waitKey()      # 키 값 대기
cv2.destroyAllWindows()    # 창 닫음