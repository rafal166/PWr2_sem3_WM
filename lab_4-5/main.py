import cv2
import numpy as np

fujiWindow = np.array([	['G', 'B', 'R', 'G', 'R', 'B'],
						['R', 'G', 'G', 'B', 'G', 'G'],
						['B', 'G', 'G', 'R', 'G', 'G'],
						['G', 'R', 'B', 'G', 'B', 'R'],
						['B', 'G', 'G', 'R', 'G', 'G'],
						['R', 'G', 'G', 'B', 'G', 'G']])


bayerWindow = np.array([['G', 'B'],
						['R', 'G']])

def przytnij(img, multipleSize):
	width = len(img)
	height = len(img[0])
	widthCrop = width % multipleSize
	heightCrop = height % multipleSize

	return img[0:(width - widthCrop), 0:(height - heightCrop), :]

def mozajkuj(img, window):
	width = len(img)
	height = len(img[0])
	windowSize = len(window)

	imgOut = img.copy()

	for i in range(0, width - windowSize, windowSize):
		for j in range(0, height - windowSize, windowSize):
			for w1 in range(windowSize):
				for w2 in range(windowSize):
					if(window[w1][w2] == 'R'):
						imgOut[i+w1][j+w2][1] = 0
						imgOut[i+w1][j+w2][2] = 0
					elif(window[w1][w2] == 'G'):
						imgOut[i+w1][j+w2][0] = 0
						imgOut[i+w1][j+w2][2] = 0
					elif(window[w1][w2] == 'B'):
						imgOut[i+w1][j+w2][0] = 0
						imgOut[i+w1][j+w2][1] = 0

	return przytnij(imgOut, windowSize)

def demozajkuj(img, window):
	width = len(img)
	height = len(img[0])
	windowSize = len(window)

	# przycinanie, żeby pasowało do maski
	imgOut = przytnij(img.copy(), windowSize)

	for i in range(0, width-windowSize, 1):
		for j in range(0, height-windowSize, 1):
			R_num = G_num = B_num = 0
			R_val = G_val = B_val = 0
			
			for w1 in range(windowSize):
				for w2 in range(windowSize):
					if(img[i+w1][j+w2][0] != 0):
						R_num += 1
						R_val += img[i+w1][j+w2][0]
					elif(img[i+w1][j+w2][1] != 0):
						G_num += 1
						G_val += img[i+w1][j+w2][1]
					elif(img[i+w1][j+w2][2] != 0):
						B_num += 1
						B_val += img[i+w1][j+w2][2]
			
			for w1 in range(windowSize):
				for w2 in range(windowSize):
					imgOut[i+w1][j+w2][0] = R_val / R_num if(R_num > 0) else 0
					imgOut[i+w1][j+w2][1] = G_val / G_num if(G_num > 0) else 0
					imgOut[i+w1][j+w2][2] = B_val / B_num if(B_num > 0) else 0

	return imgOut
			

#konfiguracja programu
windowIn = bayerWindow
nazwaPliku = 'bayer'

# windowIn = fujiWindow
# nazwaPliku = 'fuji'

# wczytanie obrazka z pliku PNG
img = cv2.imread('kot.png', cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

imgMozajkowane = mozajkuj(img, windowIn)
imgDemozajkowane = demozajkuj(imgMozajkowane, windowIn)

cv2.imwrite(nazwaPliku + '_mozaikowane_kot.png', imgMozajkowane)
cv2.imwrite(nazwaPliku + '_demozaikowane_kot.png', imgDemozajkowane)
# cv2.imshow('Wynik', imgDemozajkowane)
# cv2.waitKey(0)
# cv2.destroyAllWindows()