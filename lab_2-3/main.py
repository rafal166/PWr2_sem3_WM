import random
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.interpolate as interpolate
import time

nazwaPliku = ''

def getSinglePoint(x):
	global a, b, c, d
	return a * x**3 + b * x**2 + c * x + d

def punktyZWielomianu(num_points):
	points = []
	for i in range(0, num_points*2, 2):
		points.append((i, getSinglePoint(i)))
	return points

def splitPoints(points):
	x = np.array(points)[:, 0]
	y = np.array(points)[:, 1]
	return x, y

# moje algorytmy
def interpolacjaNajblizszySasiad(points, wbudowany = True):
	global nazwaPliku
	x, y = splitPoints(points)
	n = len(points)
	x_i = np.linspace(np.min(x)+1, np.max(x), n)

	nazwaPliku = 'najblizszy_' + ('wbudowany' if wbudowany else 'wlasny') + '_' + str(n)+'punktow'
	if(wbudowany):
		f = interpolate.interp1d(x, y, kind="nearest")
		return list(zip(x_i, f(x_i)))
	else:
		y_i = np.zeros(n)
		for i in range(n):
			idx = np.argmin(np.abs(x_i[i] - x))
			y_i[i] = y[idx]
			
		return list(zip(x_i, y_i))

def interpolacjaLiniowa(points, wbudowany = True):
	global nazwaPliku
	n = len(points)
	x, y = splitPoints(points)

	# lista xów do interpolowania
	x_i = np.linspace(np.min(x)+1, np.max(x), n)
	y_i = np.zeros(n)
	
	nazwaPliku = 'liniowa_' + ('wbudowany' if wbudowany else 'wlasny') + '_' + str(n)+'punktow'
	if(wbudowany):
		f = interpolate.interp1d(x, y, kind="linear")
		return list(zip(x_i, f(x_i)))
	else:
		for j in range(n):	
			x_out = x_i[j]
			if x_out <= points[0][0]:
				y_i[j] = points[0][1]
			elif x_out >= points[n-1][0]:
				y_i[j] = points[n-1][1]
			else:
				i = 0
				while x_out > points[i+1][0]:
					i += 1
				x1, y1 = points[i]
				x2, y2 = points[i+1]
				slope = (y2 - y1) / (x2 - x1)
				y_i[j] = y1 + slope * (x_out - x1)

		return list(zip(x_i, y_i))
	
def interpolacjaKwadratowa(points, wbudowany = True):
	global nazwaPliku
	n = len(points)
	x, y = splitPoints(points)

	# lista xów do interpolowania
	x_i = np.linspace(np.min(x)+1, np.max(x), n)
	y_i = np.zeros(n)
	
	nazwaPliku = 'kwadratowa_' + ('wbudowany' if wbudowany else 'wlasny') + '_' + str(n)+'punktow'
	if(wbudowany):
		f = interpolate.interp1d(x, y, kind="quadratic")
		return list(zip(x_i, f(x_i)))
	else:
		for j in range(n):	
			x_out = x_i[j]
			if x_out <= points[0][0]:
				y_i[j] = points[0][1]
			elif x_out >= points[n-1][0]:
				y_i[j] = points[n-1][1]
			else:
				i = 0
				while x_out > points[i+1][0]:
					i += 1
				if i == 0:
					i = 1
				elif i == n - 2:
					i = n - 3
				x0, y0 = points[i-1]
				x1, y1 = points[i]
				x2, y2 = points[i+1]
				a = (y0/(x0-x1)/(x0-x2) + y1/(x1-x0)/(x1-x2) + y2/(x2-x0)/(x2-x1))
				b = (y0/(x0-x1) + y1/(x1-x0) - a*(x0+x1))
				c = y0 - a*x0**2 - b*x0
				y_i[j] = a*x_out**2 + b*x_out + c

		return list(zip(x_i, y_i))

def interpolacjaSześcienna(points, wbudowany = True):
	global nazwaPliku
	n = len(points)
	x, y = splitPoints(points)

	# lista xów do interpolowania
	x_i = np.linspace(np.min(x)+1, np.max(x), n)
	y_i = np.zeros(n)
	
	nazwaPliku = 'szescian_' + ('wbudowany' if wbudowany else 'wlasny') + '_' + str(n)+'punktow'
	if(wbudowany):
		f = interpolate.interp1d(x, y, kind="cubic")
		return list(zip(x_i, f(x_i)))
	else:
		for j in range(n):	
			x_out = x_i[j]
			if x_out <= points[0][0]:
				y_i[j] = points[0][1]
			elif x_out >= points[n - 1][0]:
				y_i[j] = points[n - 1][1]
			else:
				i = 0
				while x_out > points[i+1][0]:
					i += 1
				if i == 0:
					i = 1
				elif i == n - 2:
					i = n - 3
				x0, y0 = points[i-1]
				x1, y1 = points[i]
				x2, y2 = points[i+1]
				x3, y3 = points[i+2]
				y_i[j] = y0 * ((x_out - x1) * (x_out - x2) * (x_out - x3)) / ((x0 - x1) * (x0 - x2) * (x0 - x3)) + y1 * ((x_out - x0) * (x_out - x2) * (x_out - x3)) / ((x1 - x0) * (x1 - x2) * (x1 - x3)) + y2 * ((x_out - x0) * (x_out - x1) * (x_out - x3)) / ((x2 - x0) * (x2 - x1) * (x2 - x3)) + y3 * ((x_out - x0) * (x_out - x1) * (x_out - x2)) / ((x3 - x0) * (x3 - x1) * (x3 - x2)) + ((x_out - x0) * (x_out - x1) * (x_out - x2) * (x_out - x3)) / ((x0 - x1) * (x0 - x2) * (x0 - x3) * (x1 - x2) * (x1 - x3) * (x2 - x3))

		return list(zip(x_i, y_i))

def bladSredniokwadratowy(points):
	n = len(points)
	error = 0
	for i in range(n):
		xi, yi = points[i]
		yr = getSinglePoint(xi)
		error += (yi - yr) ** 2
	return error / n

HOW_MUCH = 10000
# LICZYC_WBUDOWANA = False
LICZYC_WBUDOWANA = True
a= 2
b = -3
c = 1
d = 5
# interpolacja najbliższym sąsiadem 
generated = punktyZWielomianu(HOW_MUCH)

startTime = time.time()
# interpolated = interpolacjaNajblizszySasiad(generated, LICZYC_WBUDOWANA)
# interpolated = interpolacjaLiniowa(generated, LICZYC_WBUDOWANA)
# interpolated = interpolacjaKwadratowa(generated, LICZYC_WBUDOWANA)
interpolated = interpolacjaSześcienna(generated, LICZYC_WBUDOWANA)
endTime = time.time()
elapsedTime = endTime - startTime

print(f"Czas wykonania: {elapsedTime} sekund")
print(f"Błąd średniokwadratowy: {str(bladSredniokwadratowy(interpolated))}")

gfp_x, gfp_y = splitPoints(generated)
ifp_x, ifp_y = splitPoints(interpolated)

plt.plot(ifp_x, ifp_y, label="Interpolowane")
# plt.plot(gfp_x, gfp_y, label="Wygenerowane")
plt.legend()

# plt.savefig('output/' + nazwaPliku + '.png')
plt.show()
