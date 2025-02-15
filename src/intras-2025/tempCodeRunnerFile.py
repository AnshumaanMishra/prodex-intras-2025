[cls])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        lightLocation = (personLocation + lightLocation) // 2
        cv2.circle(img, lightLocation, radius=10, color=(0, 0, 255), thickness=-1)
        # object details
        org = [x1, y1]
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
        movement = (personLocation - lightLocation)
        cv2.putText(img, classNames[cls] + str(f"{personLocation, movement}"), org, font, fontScale, color, thickness)