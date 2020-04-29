import os
from utils.opencv_face_detector import detect_face


def gen_labels(images_path, label_file_path):
    if not os.path.exists(label_file_path):
        label_file = open(label_file_path, 'w')
        train_images = os.listdir(images_path)
        train_count = 0
        face_count = 0
        for i, train_image in enumerate(train_images):
            if i % 1000 == 0:
                print(i, len(train_images))
            face_coordinates = detect_face(os.path.join(images_path, train_image))
            if len(face_coordinates) == 0:
                continue

            label_file.write(os.path.join(images_path, train_image))

            image_face_bboxes = ""
            for face_coordinate in face_coordinates:
                face_count += 1
                (x, y, w, h) = face_coordinate
                image_face_bboxes += " {} {} {} {}".format(x, y, w, h)
            label_file.write("{}\n".format(image_face_bboxes))

            train_count += 1

        print("{} faces detected from {} images: ".format(face_count, train_count))
