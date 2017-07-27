import numpy as np
import cv2
img = cv2.imread('new_clipped_vh.tif', -1)   # read tiff image as it is
# img = Image.open('new_clipped_vh.tif')

img = cv2.convertScaleAbs(img)
cv2.imwrite('in_int8_new_clipped_vh.tif', img)
I = np.array(img)
print(I.dtype)
print(I.shape)


padded_I = np.lib.pad(I, 1, 'edge')
avg_I = np.zeros(I.shape, dtype=np.int)

(wi, hei) = padded_I.shape
for i in range(1, wi-1):
    for j in range(1, hei-1):
        for p in range(-1, 2):
            for q in range(-1, 2):

                avg_I[i-1, j-1] += padded_I[i+p, j+q]

        I[i-1, j-1] = int(avg_I[i-1, j-1]/9)

cv2.imwrite("avg.tif", I)


num_classes = 4
num_samples_each_class = 2000
(width, height) = padded_I.shape
# width = 200    # for testing assign smaller values
# height = 200

t = [2, 5, 9, 15, 25]  # design parameter: threshold
h = 5  # design parameter: window size
win = int((h-1)/2)
# connected component labeling


def label(s):
    # sl = np.array([[0 for p in range(h)]for q in range(h)])
    sl = np.zeros((h, h), dtype=np.int)
    l = 1
    eq = {}
    if s[0, 0] > 0:
        sl[0, 0] = l
        l += 1
    for i in range(1, h):  # for col no zero
        if s[i, 0] > 0:  # foreground
            if s[i, 0] == s[i-1, 0]:
                sl[i, 0] = sl[i-1, 0]
            else:
                sl[i, 0] = l
                l += 1

    for j in range(1, h):   # for row no zero
        if s[0, j] > 0:  # foreground

            if s[0, j] == s[0, j-1]:
                sl[0, j] = sl[0, j-1]
            else:
                sl[0, j] = l
                l += 1

    for i in range(1, h):
        for j in range(1, h):
            if s[i, j] > 0:
                if sl[i, j-1] > 0 and sl[i-1, j] > 0:  # equivalent component
                    sl[i, j] = sl[i, j-1]
                    if sl[i, j-1] != sl[i-1, j]:
                        eq[min(sl[i, j - 1], sl[i - 1, j])] = max(sl[i, j - 1], sl[i - 1, j])
                        # print(eq)
                elif s[i, j] == s[i-1, j]:
                    sl[i, j] = sl[i-1, j]
                elif s[i, j] == s[i, j-1]:
                    sl[i, j] = sl[i, j-1]
                else:
                    sl[i, j] = l
                    l += 1

    for k, v in sorted(eq.items()):
        sl[sl == min(k, v)] = max(k, v)
    '''
    err = {}    
    for i in range(1, h):
        for j in range(1, h):
            if sl[i, j] > 0 and sl[i-1][j] >0:
                if sl[i, j] != sl[i-1, j]:
                    err[]
    '''

    # print(eq)
    return sl

# Function to find sub histograms
def subhist(sl):

    shist = np.zeros(6, dtype=int)
    spam = np.unique(sl, return_counts=True)
    count = spam[1]

    if spam[0][0] == 0:
        count = np.delete(count, 0)

    # print('spam ', spam)
    # print('count ', count)
    hist = np.bincount(count)

    # print(hist)
    try:
        shist[0] = 0
        shist[1] = hist[1]
        shist[2] = np.sum(hist[2:4])
        shist[3] = np.sum(hist[4:9])
        shist[4] = np.sum(hist[9:16])
        shist[5] = np.sum(hist[16:])
    except:
        pass
    '''
    for k in range(2, 6):
        try:
            mask = [range((k-1)**2, (k**2-1))]
            hhh = hist[(k-1)**2:(k**2)]
            shist[k] = np.sum(hhh)
            print(mask)
            print(k, hhh)
        except:
            continue
            '''
    # print(shist)
    return shist

# LPH = np.zeros((width*height, 16),  dtype=int)

'''
Function to find local pattern histogram (LPH)
'''


def lph(x, y):     # inserted two arguments

    lpdata = []

    sp = np.zeros((h, h), dtype=np.int)
    se = np.zeros((h, h), dtype=np.int)
    sn = np.zeros((h, h), dtype=np.int)
    # for x in range(win, width-win):
    for i in range(0, len(x)):
        # for y in range(win, height-win):

        lphist = []
        for j in range(len(t)):
            for hw in range(-win, win+1):
                for hh in range(-win, win+1):

                    if I[x[i]+hw, y[i]+hh] > (I[x[i], y[i]]+t[j]):
                        sp[hw+win][hh+win] = 1
                    elif (I[x[i], y[i]]-t[j]) <= I[x[i]+hw, y[i]+hh] <= (I[x[i], y[i]]+t[j]):
                        se[hw+win][hh+win] = 1
                    elif I[x[i]+hw, y[i]+hh] < (I[x[i], y[i]]-t[j]):
                        sn[hw+win][hh+win] = 1
            # print('sp', sp)
            # print('se', se)
            # print('sn', sn)
            sp = label(sp)
            se = label(se)
            sn = label(sn)
            # print('\nsp label', sp)
            # print('se label', se)
            # print('sn label', sn)

            lphist += list(np.delete(subhist(sp), 0)) + list(np.delete(subhist(se), 0)) + list(np.delete(subhist(sn), 0))

            # print(lphist)  # This is pixel by pixel feature .....want to store this in numpy array
            sp.fill(0)
            se.fill(0)
            sn.fill(0)

        lpdata.append(lphist)

    featuredata = np.vstack(lpdata)
    return featuredata


''' Read Ground truth images for training
5 classes and labels are given below
lw -> water ->1
# lf -> flood plane -> 
li -> irrigation -> 2
lv -> vegetation -> 3
lu -> urban -> 4
'''

lw = cv2.imread("new_clipped_water.png", 0)
# lf = cv2.imread("new_clipped_floodplain.png", 0)
li = cv2.imread("new_clipped_irrigation.png", 0)
lv = cv2.imread("new_clipped_vegetation.png", 0)
lu = cv2.imread("new_clipped_urban.png", 0)

true_data = np.zeros(num_samples_each_class*num_classes, dtype=int)
# true_data = np.zeros(num_samples_each_class*5, dtype=int)
true_data[0:num_samples_each_class] = 1
true_data[1*num_samples_each_class:2*num_samples_each_class] = 2
true_data[2*num_samples_each_class:3*num_samples_each_class] = 3
true_data[3*num_samples_each_class:4*num_samples_each_class] = 4
# true_data[4*num_samples_each_class:5*num_samples_each_class] = 5
# print(true_data)
'''
print(lw.shape)
print(np.unique(lw, return_counts=True))

print(lf.shape)
print(np.unique(lf, return_counts=True))
print(li.shape)
print(np.unique(li, return_counts=True))
print(lv.shape)
print(np.unique(lv, return_counts=True))
print(lu.shape)
print(np.unique(lu, return_counts=True))
'''
def get_sample_index(image):
    coordinate = np.argwhere(image)
    np.random.shuffle(coordinate)
    return coordinate[0:num_samples_each_class, 0], coordinate[0:num_samples_each_class, 1]

index_x = np.zeros(num_samples_each_class*num_classes, dtype=int)
index_y = np.zeros(num_samples_each_class*num_classes, dtype=int)

# for water(1)
index_x[0:num_samples_each_class], index_y[0:num_samples_each_class] = get_sample_index(lw)
# for flood plane(2)
index_x[1*num_samples_each_class:2*num_samples_each_class], index_y[1*num_samples_each_class:2*num_samples_each_class] = get_sample_index(li)

# for irrigation(3)
index_x[2*num_samples_each_class:3*num_samples_each_class], index_y[2*num_samples_each_class:3*num_samples_each_class] = get_sample_index(lv)

# for vegetation(4)
index_x[3*num_samples_each_class:4*num_samples_each_class], index_y[3*num_samples_each_class:4*num_samples_each_class] = get_sample_index(lu)

# for urban(5)
# index_x[4*num_samples_each_class:5*num_samples_each_class], index_y[4*num_samples_each_class:5*num_samples_each_class] = get_sample_index(lu)

# print(index_x)
# print(index_y)
# print(len(index_x))

LPH = lph(index_x, index_y)    # lph should be calculated for 100 coordinates

# print(LPH)
print('training data size = ', LPH.shape)
np.save('train_features', LPH)
np.save('target_class', true_data)

cv2.waitKey(0)
