from identification import BooksImage, findBook
from label_tracking import center_spine
import argparse


def alignment(lcc_code, camera_idx=0):
    booksimg = BooksImage(camera_idx=camera_idx)
    booksimg.preprocessAndReadLabels()
    label_rectangle = findBook(booksimg, lcc_code)
    if label_rectangle is not None:
        center_spine(label_rectangle, camera_idx)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-l", "--label", type=str, help="Label to track")
    args = vars(ap.parse_args())
    label = args["label"]
    if label is None:
        with open('label.txt', 'r') as f:
            label = f.readline().replace('\n', '')
    alignment(label, 1)
