from identification import BooksImage, findBook
from label_tracking import center_spine
from move_robot import MoveRobot
from book_match import label_codes_original
import argparse
from time import sleep


def processImage(camera_idx=0, all_labels=label_codes_original):
    booksimg = BooksImage(camera_idx=camera_idx)
    booksimg.preprocessAndReadLabels(all_labels)
    return booksimg


def alignment(lcc_code, camera_idx=0, num_find_attempts=5, all_labels=label_codes_original):
    booksimg = processImage(camera_idx, all_labels)
    mv = MoveRobot()
    label_rectangle = findBook(booksimg, lcc_code)
    count_failures = 0
    while (label_rectangle is None):
        count_failures += 1
        if count_failures < num_find_attempts:
            print('Moving forward and trying again...')
        else:
            print('Failed to find book after ' + str(count_failures) + 'attempts')
            return
        mv.setSpeed(0.05)
        sleep(1)
        mv.setSpeed(0)
        booksimg = processImage(camera_idx)
        label_rectangle = findBook(booksimg, lcc_code)
    res = center_spine(label_rectangle, camera_idx, debug=True)
    mv.shutDown()
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-l", "--label", type=str, help="Label to track")
    args = vars(ap.parse_args())
    label = args["label"]
    if label is None:
        with open('label.txt', 'r') as f:
            label = f.readline().replace('\n', '')
    alignment(label, 0)
