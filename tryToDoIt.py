
from huffman2 import HuffmanCoding
import sys
import numpy
from PIL import *
from PIL import Image, ImageTk
import numpy as np
import os
from numpy import ndarray
import Huffman
import math




def main():
    #
    # # img = readPILimg("red.png")
    # # arr = PIL2np(img)
    # # print(arr)
    # arr = [[21,21,21,95,169,243,243,243],
    #        [21,21,21,95,169,243,243,243],
    #        [25,0,21,95,169,243,243,243],
    #        [21,21,21,95,169,243,243,23]]

    # Darr_arr = difference(arr)
    # print(Darr_arr)
    # print(len(Darr_arr[0]))
    # print(reDifference(Darr_arr))
    # #RGB_comp("muhi.png")
    # # img = readPILimg("muhi.png")
    # # arr = PIL2np(img)

    # Dosya açar ve içine veri aktarır.
    # with open("gry_ımg.txt", "w+") as file:
    #     pass
    # with open("gry_ımg.txt", "r+") as file:
    #     writeMatrixToFile(file, arr)

    # array =readFileTo2DArray("gry_ımg.txt")
    # array2 = readFileTo2DArray("blue_diff_encode.txt")
    # print(difference(array))
    # print(difference(array2))
     #Level1_HuffmanEncodingAndDecoding("input.txt")
    #Level2_ImageCompressionGrayLevel("uncompressed_cat2.png", "muhi2.txt", "Level2_restoredImage.png")
    #Level3_ImageCompressionGrayLeveldifferences("uncompressed_cat.png", "test_encode.txt", "restored_image.png")
    #Level4ImageCompression("uncompressed_cat2.png")
    Level5ImageCompressionColordifferences("uncompressed_cat2.png")
    # Level5("muhi.png")
    # print(comparison("muhi.png","restored_image.png"))
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


def Level1_HuffmanEncodingAndDecoding(file):
    path = file
    h = HuffmanCoding(path)

    with open(file, "r+") as f:
        file_data = f.read()

    encoding, tree = Huffman.Huffman_Encoding(file_data)
    print("Encoded output", encoding)
    with open("Level1_test.txt", "w+") as f:
        f.write(encoding)

    output_path = h.compress()
    print("Compressed file path: " + output_path)

    # text to str.
    fileBinary = open("Level1_test.txt", "r")
    encode_str = fileBinary.read()
    fileBinary.close()

    decom_path = h.decompress(output_path)
    print("Decompressed file path: " + decom_path)
    decodedOutput = Huffman.Huffman_Decoding(encode_str, tree)
    print("Decoded Output", decodedOutput.replace(" ", ""))


def Level2_ImageCompressionGrayLevel(ımg, Img_encode, restored_ımage):
    img = readPILimg(ımg)
    arr = PIL2np(img)

    # Dosya açar ve içine veri aktarır.
    with open("gry_ımg.txt", "w+") as file:
        pass
    with open("gry_ımg.txt", "r+") as file:
        writeMatrixToFile(file, arr)

    print("COMPRESSION")
    encoding, tree = Huffman.Huffman_Encoding(readFileToList("gry_ımg.txt"))
    print("Encoded output", encoding)

    # bu arrayden değerler okunacak
    gray2DArray = arr
    huffman_encoding = Huffman.Calculate_Codes(tree)

    # bu arraye binary kodlar yazılacak
    bin_encoding = []
    for i in range(len(gray2DArray)):
        colum = []
        for j in range(len(gray2DArray[0])):
            colum.append((huffman_encoding[gray2DArray[i][j]]))
        bin_encoding.append(colum)

    with open(Img_encode, "w+") as a:
        writeMatrixToFile(a, bin_encoding)

    path = Img_encode
    h = HuffmanCoding(path)

    output_path = h.compress()
    print("Compressed file path: " + output_path)

    print("DECOMPRESSION")

    decom_path = h.decompress(output_path)
    print("Decompressed file path: " + decom_path)

    ArrayBin = readFileTo2DArray(decom_path, str)
    row = (len(ArrayBin))
    column = (len(ArrayBin[0]))
    # text to str.
    bin = open(decom_path)
    stringEncode = bin.read()
    bin.close()

    decom_path = np.array(decom_path, copy=True)

    print("Decoded Output", Huffman.Huffman_Decoding(stringEncode, tree))
    strDecode = Huffman.Huffman_Decoding(stringEncode, tree)

    decode_arr = np.fromstring(strDecode, dtype=int, sep=' ')
    lastarr = decode_arr.reshape(row ,column)

    convertMatrixToImage(lastarr, restored_ımage)


def Level3_ImageCompressionGrayLeveldifferences(image, diff_encode, restoredImage):
    lev_3Img = readPILimg(image)
    ImgNpArray = PIL2np(lev_3Img)

    ImgArray = Array2DToList2D(ImgNpArray)
    difference_array = difference(ImgArray)
    difference_array = np.array(difference_array)
    print("difference pixels :", difference(difference_array))

    # Dosya açar ve içine veri aktarır.
    with open("Level3_diff.txt", "w+") as file:
        pass
    with open("Level3_diff.txt", "r+") as file:
        writeMatrixToFile(file, difference_array)

    print("------------------------------------------------COMPRESSION")
    encoding, tree = Huffman.Huffman_Encoding(readFileToList("Level3_diff.txt"))
    print("Encoded output", encoding)

    # bu arrayden değerler okunacak
    difference_2DArray = difference_array
    huff_encode = Huffman.Calculate_Codes(tree)

    # bu arraye binary kodlar yazılacak
    binEncode = []
    for ind in range(len(difference_2DArray)):
        column = []
        for ind2 in range(len(difference_2DArray[0])):
            column.append((huff_encode[difference_2DArray[ind][ind2]]))
        binEncode.append(column)

    with open(diff_encode, "w+") as f:
        writeMatrixToFile(f, binEncode)

    path = diff_encode
    h = HuffmanCoding(path)
    output_path = h.compress()
    print("Compressed file path: " + output_path)

    print("DECOMPRESSION")

    decom_path = h.decompress(output_path)
    print("Decompressed file path: " + decom_path)

    decompressfile = readFileTo2DArray(decom_path, str)

    Row = (len(decompressfile))
    Colmn = (len(decompressfile[0]))
    # text to str.
    binfile = open(decom_path)
    encode = binfile.read()
    binfile.close()

    decom_path = np.array(decom_path, copy=True)

    print("Decoded Output", Huffman.Huffman_Decoding(encode, tree))
    decode_str = Huffman.Huffman_Decoding(encode, tree)

    decode_arr = np.fromstring(decode_str, dtype=int, sep=' ')
    decode_array = decode_arr.reshape(Row, Colmn)
    print("-----------")
    print(decode_array)

    original = reDifference(decode_array)

    convertMatrixToImage(original, restoredImage)

def Level4ImageCompression(img):

    RGB_comp(img, 'blue', "blue.png")
    RGB_comp(img, 'green', "green.png")
    RGB_comp(img, "red", "red.png")

    print(" ")
    print(" ")

    print("Blue Img:")
    Level2_ImageCompressionGrayLevel(ımg="blue.png",
                                     Img_encode="bluencode.txt",
                                     restored_ımage="bluerestoredImage.png")

    print("Green Img:")
    Level2_ImageCompressionGrayLevel(ımg="green.png",
                                     Img_encode="greenencode.txt",
                                     restored_ımage="greenrestoredImage.png")


    print("Red Img:")
    Level2_ImageCompressionGrayLevel(ımg="red.png",
                                     Img_encode="redencode.txt",
                                     restored_ımage="redrestoredImage.png")


def Level5ImageCompressionColordifferences(img):

    RGB_comp(img, 'blue', "blue.png")
    RGB_comp(img, 'green', "green.png")
    RGB_comp(img, "red", "red.png")

    print("For Blue Img:")
    Level3_ImageCompressionGrayLeveldifferences(image="blue.png",
                                                diff_encode="blue_diff_encode.txt",
                                                restoredImage="blue_restored_DiffImage.png")

    print(" ")
    print(" ")

    print("For Green Img:")
    Level3_ImageCompressionGrayLeveldifferences(image="green.png",
                                                diff_encode="green_diff_encode.txt",
                                                restoredImage="green_restored_DiffImage.png")

    print(" ")
    print(" ")

    print("For Red Img:")
    Level3_ImageCompressionGrayLeveldifferences(image="red.png",
                                                diff_encode="red_diff_encode.txt",
                                                restoredImage="red_restored_DiffImage.png")


def Array2DToList2D(npArray):
    list_of_lists = list()
    for row in npArray:
        list_of_lists.append(row.tolist())

    return list_of_lists


def difference(arr):
    # en boy
    nrow = len(arr)
    ncolumn = len(arr[0])


    pivot_arr = []
    for i in range(nrow):
        pivot_arr.append(arr[i][0])
    with open("pivot.txt", "w+" )as f:
        pass
    with open("pivot.txt", "r+") as f:

        writeArrToFile(f, pivot_arr)

    darr = numpy.array(arr, copy=True)
    for i in range(nrow):
         for j in range(1, ncolumn):
            darr[i][j] = arr[i][j] - arr[i][j - 1]


    diff_arr = numpy.array(darr, copy=True)
    pivot = darr[0][0]
    diff_arr[0][0] = darr[0][0] - pivot
    for i in range(1, nrow):
        diff_arr[i][0] = (darr[i][0]) - int(darr[i - 1][0])


    return diff_arr


def reDifference(diff_arr):

    nrow = len(diff_arr)
    ncolumn = len(diff_arr[0])
    pivot = []

    with open("pivot.txt", "r") as f:
        str_arr = ','.join([l.strip() for l in f])

    piv_arr = np.asarray(str_arr.split(' '), dtype=int)

    darr = np.array(diff_arr, copy=True)

    for i in range(0, nrow):
        darr[i][0] = piv_arr[i]
    print()
    arr = np.array(darr, copy=True)


    for ind in range(nrow):
        for j in range(1, ncolumn):
            arr[ind][j] = darr[ind][j] + arr[ind][ j -1]

    return arr


def Array2D_To_Array1D(arr2D):
    arr1D = []
    for ind in range(0, len(arr2D)):
        for ind2 in range(0, len(arr2D[0])):
            arr1D.append(arr2D[ind][ind2])
    return arr1D


def convertMatrixToImage(arr, str):
    # convert 2D array to image
    array = np2PIL(arr)
    array.save(str)


def pil_to_np(img):
   img_array = np.array(img)
   return img_array


def my_function(filepath):
    data = open(filepath, "r+")


def strToArr2DWithSpace(str, row, colmn):
    arr = np.fromstring(str, sep=" ", dtype=int)
    arr2D = np.reshape(arr, (colmn, row))
    return arr2D


def ListToNpArray(list):
    arr = numpy.array(list)
    return arr


def stringToArray2DForImg(w, h, str):
    arr2D = [w][h]
    arr1D = readFileTo1DArray(str)
    allPixels = 0
    while allPixels <= h + w:
        for ind in range(0, h):
            for ind2 in range(0, w):
                arr2D[ind][ind2] = arr1D[allPixels]
                allPixels += 1
    return arr2D


def Array2DToText(arr):
    # file_data = np.loadtxt(arr, dtype=int)
    file_data = arr
    text = ""
    for val in range(0, len(arr)):
        for val2 in range(0, len(arr[0])):
            text = text + ' ' + str(file_data[val][val2])

    return text


def saveBinFile(str):
    f = open("compressed_file.bin", "wb+")
    textToArray = np.array(list(str))
    arr = bytearray(textToArray)
    f.write(arr)
    f.close()


def readBinFile(binFile):
    with open(binFile, "br") as f:
        buffer = f.read()
        print("Length of buffer is %d" % len(buffer))

        for i in buffer:
            print(int(i))


def readFileTo1DArray(file):
    file_data = np.loadtxt(file, dtype=int)
    return file_data


# boşluklu txt yi listeye çevirir.
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


def readFileTo2DArray(file, type):
    if type == int:
        file_data = np.loadtxt(file, dtype=type)
    else:
        file_data = np.loadtxt(file, dtype=str)
    return file_data

def writeArrToFile(file, arr):
    # input dosyasına data yazdırır.
    for ind in range(0, len(arr)):
        file.write(' ' + str(arr[ind]))


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
    # img_gray.save("newImg", 'png')
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


def RGB_comp(image_file_path, channel ,str):
   # red channel -> 0, green channel -> 1 and blue channel -> 2
   if channel == 'red':
      channel_index = 0
   elif channel == 'green':
      channel_index = 1
   else:
      channel_index = 2
   # open the current image as a PIL image
   img_rgb = Image.open(image_file_path)
   # convert the current image to a numpy array
   image_array = pil_to_np(img_rgb)
   # traverse all the pixels in the image array
   n_rows = image_array.shape[0]
   n_cols = image_array.shape[1]
   for row in range(n_rows):
      for col in range(n_cols):
         # make all the values 0 for the color channels except the given channel
         for rgb in range(3):
            if (rgb != channel_index):
               image_array[row][col][rgb] = 0
   # convert the modified image array (numpy) to a PIL image
   pil_img = np_to_pil(image_array)
   pil_img.save(str)


def np_to_pil(img_array):
   img = Image.fromarray(np.uint8(img_array))
   return img

def open_ready_file(file):
    f = open(file, "w+")
    f = open(file, "r+")

if __name__ == '__main__':
    main()
