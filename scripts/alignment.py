from identification import BooksImage, findBook
from label_tracking import controller_center_book


def alignment(lcc_code, camera_idx=0):
    booksimg = BooksImage(webcam_idx=camera_idx)
    booksimg.preprocessAndReadLabels()
    label_rectangle = findBook(booksimg, lcc_code)
    controller_center_book(label_rectangle, camera_idx)
