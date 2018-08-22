import cv2



image=cv2.imread("/home/andrei/temp/test/test_1.jpg")

if image is None:
    print("Este none")

print(image)

print(image.shape)