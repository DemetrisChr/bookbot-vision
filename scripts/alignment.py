from identification import BooksImage, findBook
from label_tracking import center_spine


def alignment(lcc_code, camera_idx=0):
    booksimg = BooksImage(camera_idx=camera_idx)
    booksimg.preprocessAndReadLabels()
    label_rectangle = findBook(booksimg, lcc_code)
    if label_rectangle is not None:
        center_spine(label_rectangle, camera_idx)


if __name__ == '__main__':
    lcc_codes = ['DG311 Gib.', 'BJ1499.S5 Kag.', 'QC21.3 Hal.', 'QC174.12 Bra.', 'PS3562.E353 Lee.',
                 'PR4662 Eli.', 'HA29 Huf.', 'QA276 Whe.', 'QA76.73.H37 Lip.', 'QA76.62 Bir.']
    alignment(lcc_codes[9], 1)
