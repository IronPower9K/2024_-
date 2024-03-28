import cv2 as cv
import sys
import numpy as np






# 웹 캠에서 비디오 읽기
cap = cv.VideoCapture(0)

if not cap.isOpened():
    sys.exit("카메라 연결 실패")

while True:
    ret, frame = cap.read()

    if not ret:
        print("프레임 획득에 실패하여 루프를 나감")
        break

    cv.imshow("video display", frame)

    key = cv.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv.destroyAllWindows()


'''
# 비디오에서 수집한 영상 이어 붙이기
cap = cv.VideoCapture(0, cv.CAP_DSHOW)

if not cap.isOpened():
    sys.exit("카메라 연결 실패")

frames = []

while True:
    ret, frame = cap.read()

    if not ret:
        print("프레임 획득에 실패하여 루프를 나감")
        break

    cv.imshow("video display", frame)

    key = cv.waitKey(1)
    if key == ord("c"):
        frames.append(frame)

    elif key == ord("q"):
        break

cap.release()
cv.destroyAllWindows()

if len(frames) > 0:
    imgs = frames[0]
    for i in range(1, min(3, len(frames))):
        img = np.hstack((imgs, frames[i]))
    
    cv.imshow("collected images", imgs)

    cv.waitKey()
    cv.destroyAllWindows()

print(len(frames))
print(frames[0].shape)
print(type(imgs))
print(imgs.shape)
'''

'''
# 영상에 도형을 그리고 글씨 쓰기 1
img = cv.imread("ch2\movemin.jpg")

if img is None:
     sys.exit("파일을 찾을 수 없습니다.")

cv.rectangle(img, (290,780), (620,950), (0,0,255), 2)
cv.putText(img, "mouse", (290,770), cv.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)

cv.imshow("Draw", img)

cv.waitKey()
cv.destroyAllWindows()
'''

'''
# 영상에 도형을 그리고 글씨 쓰기 2
img = cv.imread("ch2\movemin.jpg")

if img is None:
    sys.exit("파일을 찾을 수 없습니다.")

def draw(event, x ,y, flags, param):
 
    if event == cv.EVENT_LBUTTONDOWN:
          cv.rectangle(img, (x,y), (x+200, y+200), (0,0,255), 2)

    elif event == cv.EVENT_RBUTTONDOWN:
          cv.rectangle(img, (x,y), (x+100, y+100), (255,0,0), 2)

    cv.imshow("Drawing", img)

cv.namedWindow("Drawing")
cv.imshow("Drawing", img)

cv.setMouseCallback("Drawing", draw)

while(True):
    if cv.waitKey(1)==ord("q"):
        cv.destroyAllWindows()
        break
'''
'''
# 페인팅 기능 만들기 
BrushSize = 5
LColor, RColor = (255, 0, 0), (0, 0, 255)

def paintitng(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(img, (x,y), BrushSize, LColor, -1)
    elif event == cv.EVENT_RBUTTONDOWN:
        cv.circle(img, (x,y), BrushSize, RColor, -1)
'''    