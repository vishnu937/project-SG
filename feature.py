import numpy as np
import cv2
import pickle


# img = cv2.imread("Rose.jpg", -1)

img = cv2.imread('new_clipped_vh.tif', -1)   # read tiff image as it is
I = np.array(img)
# print(I.dtype)
print(I.shape)

t = 4  # design parameter: threshold
h = 5  # design parameter: window size

# (width, height) = I.shape
width = 20    # for testing assign smaller values
height = 10

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
    while x:
        #for y in range(win, height-win):
        while y:

            for hw in range(-win, win+1):
                for hh in range(-win, win+1):

                    if I[x+hw, y+hh] > (I[x, y]+t):
                        sp[hw+win][hh+win] = 1
                    elif (I[x, y]-t) <= I[x+hw, y+hh] <= (I[x, y]+t):
                        se[hw+win][hh+win] = 1
                    elif I[x+hw, y+hh] < (I[x, y]-t):
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

            lphist = list(subhist(sp)) + list(np.delete(subhist(se), 0)) + list(np.delete(subhist(sn), 0))
            lpdata.append(lphist)
            # print(lphist)  # This is pixel by pixel feature .....want to store this in numpy array
            sp.fill(0)
            se.fill(0)
            sn.fill(0)

    featuredata = np.vstack(lpdata)
    return featuredata
    # return lpdata

# LPH = lph()
# print(LPH.shape)
# cv2.imshow('image', img)
# print(LPH)  # it should print  features corresponding to all pixels
# saving features in pickle
'''
with open('mlph_features.dat', "wb") as f:
    pickle.dump(LPH, f)

with open('mlph_features.dat', 'rb') as f:
    print(pickle.load(f))

'''
''' Read Ground truth images for training
5 classes and labels are given below
lw -> water ->1
lf -> flood plane -> 2
li -> irrigation -> 3
lv -> vegetation -> 4
lu -> urban -> 5
'''

lw = cv2.imread("new_clipped_water.png", 0)
lf = cv2.imread("new_clipped_floodplain.png", 0)
li = cv2.imread("new_clipped_irrigation.png", 0)
lv = cv2.imread("new_clipped_vegetation.png", 0)
lu = cv2.imread("new_clipped_urban.png", 0)
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

index = np.argwhere(lw)
# print(coordinates.shape)
# print(index[:10])
np.random.shuffle(index)
# print(index[0:10])

# print(lw[index[0:100, 0], index[0:100, 1]])
# print(type(index))
x = index[0:10, 0]
y = index[0:10, 1]
print(x)
print(y)
# LPH = lph(x, y)    # lph should be calculated for 100 coordinates
# print(LPH)
# print(LPH.shape)
cv2.waitKey(0)
