import fileinput
from pathlib import Path

import numpy
from PIL import *
from PIL import Image
import numpy as np
import os
from numpy import ndarray
import Huffman
import math
import cv2


def main():
    img = readPILimg("muhi.png")
    arr = PIL2np(img)
    print("---------")
    with open("input.txt", "w+") as file:
        pass
    with open("input.txt", "r+") as file:
        writeMatrixToFile(file, arr)

    print("compressionLevel1 :")
    print("---------")
    compressionLevel2("input.txt", img)
    # gray_level_list = readFileToList("input.txt")
    #
    # array = ListToNpArray(gray_level_list)
    #
    # my_filter = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    # im_out = convolve(arr, my_filter)  # 3. foto çıkıyor arrayı burası.
    # print("Photo 4:")
    # print(im_out)
    #
    # im_out = threshold(im_out, 60, 0, 100)
    # new_img = np2PIL(im_out)
    # new_img.show()


def compressionLevel2(file, img):
    nrows = img.size[0]
    ncols = img.size[1]

    encoding, tree = Huffman.Huffman_Encoding(readFileToList(file))
    print("Encoded output", encoding)

    # saves binary code in new file.
    with open("compressed_file.txt", "w+") as f:
        f.write(encoding)

    # saves binary codes ftrom new file to huffman decodding.
    compressedFile = open("compressed_file.txt", "r+")
    newEncoding = compressedFile.read()
    print("Decoded Output", Huffman.Huffman_Decoding(newEncoding, tree))

    # yeni string decodeyi iki boyutlu araya atar.
    newImgArr = strToArr2DWithSpace(Huffman.Huffman_Decoding(newEncoding, tree), nrows, ncols)
    print(newImgArr)
    # convertMatrixToImage(newImgArr)
    # stringToArray2DForImg(14, 12, newImgArr)
    convertMatrixToImage(newImgArr, "restoredImage.png")
    # -----------------------------------------------Level 3

    level3Img = readPILimg("muhi.png")
    level3ImgArray = PIL2np(level3Img)
    print(level3ImgArray)

    diff_arr = difference(newImgArr)
    print(diff_arr)
    print("difference pixels :", difference(newImgArr))

    with open("Level3.txt", "w+") as diff_file:
        writeMatrixToFile(diff_file, diff_arr)
    with open("Level3.txt", "r+") as diff_file:
        diff_encoding, diff_tree = Huffman.Huffman_Encoding(readFileToList("Level3.txt"))
        print("Encoded output", diff_encoding)
        print("Decoded Output", Huffman.Huffman_Decoding(diff_encoding, diff_tree))
    os.remove("Level3.txt")

    # --------------------------------------------


def difference(arr):
    nrow = len(arr)
    ncolumn = len(arr[0])

    darr = [[0] * ncolumn] * nrow
    for i in range(nrow):
        for j in range(1, ncolumn):
            darr[i][j] = arr[i][j] - arr[i][j - 1]

    diff_arr = darr
    pivot = darr[0][0]
    diff_arr[0][0] = darr[0][0] - pivot
    for i in range(1, nrow):
        diff_arr[i][0] = int(darr[i][0]) - int(darr[i - 1][0])
    # diff_arr = np.matrix(diff_arr)

    return diff_arr


def Array2D_To_Array1D(arr2D):
    arr1D = []
    for ind in range(0, len(arr2D)):
        for ind2 in range(0, len(arr2D[0])):
            arr1D.append(arr2D[ind][ind2])
    return arr1D


def convertMatrixToImage(arr, str):
    # convert 2D array to image
    cv2.imwrite(str, arr)


def my_function(filepath):
    data = open(filepath, "r+")


def strToArr2DWithSpace(str, row, colmn):
    arr = np.fromstring(str, sep=" ", dtype=int)
    arr2D = np.reshape(arr, (colmn, row))
    return arr2D


def ListToNpArray(list):
    arr = numpy.array(list)
    return arr


def stringToArray2DForImg(h, w, str):
    arr2D = [w][h]
    arr1D = readFileTo1DArray(str)
    allPixels = 0
    while allPixels <= h + w:
        for ind in range(0, h):
            for ind2 in range(0, w):
                arr2D[ind][ind2] = arr1D[allPixels]
                allPixels += 1
    return print(arr2D)


def Array2DToText(file):
    file_data = np.loadtxt(file, dtype=int)

    text = ""
    for val in range(0, 14):
        for val2 in range(0, 12):
            text = text + ' ' + str(file_data[val][val2])

    return text


def saveBinFile(str):
    # f = open("compressed_file.bin", "wb+")
    # textToArray = np.array(list(str))
    # arr = bytearray(textToArray)
    # f.write(arr)
    # f.close()
    pass


def readBinFile(binFile):
    with open(binFile, "br") as f:
        buffer = f.read()
        print("Length of buffer is %d" % len(buffer))

        for i in buffer:
            print(int(i))


def readFileTo1DArray(file):
    file_data = np.loadtxt(file, dtype=int)
    return file_data


def readFileToList(file):
    # opening the file in read mode
    my_file = open(file, "r")
    # reading the file
    data = my_file.read()
    # replacing end splitting the text
    # when newline ('\n') is seen.
    data_into_list = data.split(" ")
    del data_into_list[0]
    data_into_list = list(map(int, data_into_list))
    my_file.close()
    return data_into_list


def calculate_entropy(arr):
    entropyVal = 0
    for ind in range(0, len(arr)):
        entropyVal += arr[ind] * math.log(arr[ind], 2)
    return entropyVal * (-1)


def readFileTo2DArray(file):
    file_data = np.loadtxt(file, dtype=int)
    return file_data


def writeMatrixToFile(file, arr):
    # input dosyasına data yazdırır.
    for ind in range(0, len(arr)):
        if ind != 0:
            file.write("\n")
        for ind2 in range(0, len(arr[0])):
            file.write(' ' + str(arr[ind][ind2]))


def readFile(fileName):
    fileObj = open("input.txt", "r")  # opens the file in read mode
    words = fileObj.read().splitlines()  # puts the file into an array
    fileObj.close()
    return words


def readPILimg(image):
    # img dosyasının konumunu belirler.
    img = Image.open(image)
    # dosyanın içeriğini gösterir.
    img.show()
    # look at the function.
    img_gray = color2gray(img)
    print(img_gray)
    img_gray.show()
    #img_gray.save("newImg", 'png')
    # Return a resized copy of this img.
    # new_img = img.resize((256,256))
    # new_img.show()
    return img_gray


# converte the file
def color2gray(img):
    # returns converted copy of this image.For the "P" mode,
    # this method translates pixels through the palette.
    img_gray = img.convert('L')
    return img_gray


# arraya kaydetme
def PIL2np(img):
    # img.size yi dizi gibi düşün ilk elemanı[0] rows,
    # ikinci elemanı [1] column
    nrows = img.size[0]
    ncols = img.size[1]
    print("nrows, ncols : ", nrows, ncols)
    # array oluşturup değerleri içeri alır 2 D
    imgarray = np.array(img.convert("L"))
    return imgarray


def np2PIL(im):
    print("size of arr: ", im.shape)
    img = Image.fromarray(np.uint8(im))
    return img


def convolve(im, filter):
    (nrows, ncols) = im.shape
    (k1, k2) = filter.shape
    k1h = int((k1 - 1) / 2)
    k2h = int((k2 - 1) / 2)
    # np.zeros() Return a new array of given shape and type, filled with zeros.
    im_out = np.zeros(shape=im.shape)
    print("image size , filter size ", nrows, ncols, k1, k2)
    for i in range(1, nrows - 1):
        for j in range(1, ncols - 1):
            sum = 0.
            for l in range(-k1h, k1h + 1):
                for m in range(-k2h, k2h + 1):
                    sum += im[i - l][j - m] * filter[l][m]
            im_out[i][j] = sum
    return im_out


def threshold(im, T, LOW, HIGH):
    (nrows, ncols) = im.shape
    for i in range(nrows):
        for j in range(ncols):
            if abs(im[i][j]) < T:
                im[i][j] = LOW
            else:
                im[i][j] = HIGH
    return im


if __name__ == '__main__':
    main()
