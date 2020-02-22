from identification import BooksImage, findBook
from label_tracking import center_spine


def alignment(lcc_code, camera_idx=0):
    booksimg = BooksImage(camera_idx=camera_idx)
    booksimg.preprocessAndReadLabels()
    label_rectangle = findBook(booksimg, lcc_code)
    center_spine(label_rectangle, camera_idx)


if __name__ == '__main__':
    lcc_codes = ['DG311 Gib.', 'QC174.12 Bra.']
    alignment(lcc_codes[1], 1)
