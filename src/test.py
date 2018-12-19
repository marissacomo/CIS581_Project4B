import cv2

matches = []

count = 0
for p in range(0, 5):
    matches.append(cv2.DMatch(count, count, 0))
    count = count + 1

print(matches)