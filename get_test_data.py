import numpy as np
import cv2

# img = cv2.imread('new_clipped_vh.tif', -1)   # read tiff image as it is
img = cv2.imread('avg.tif', -1)
I = np.array(img)
print(I.dtype)
print(I.shape)

t = [2, 5, 9, 15, 25]  # design parameter: threshold
h = 5  # design parameter: window size
(width, height) = I.shape
# width = 10    # for testing assign smaller values
# height = 2255

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

    # print(shist)
    return shist

# LPH = np.zeros((width*height, 16),  dtype=int)

'''
Function to find local pattern histogram (LPH)
'''


def lph():     # inserted two arguments

    lpdata = []

    sp = np.zeros((h, h), dtype=np.int)
    se = np.zeros((h, h), dtype=np.int)
    sn = np.zeros((h, h), dtype=np.int)
    for x in range(win, width-win):
    # for i in range(0, len(x)):
        for y in range(win, height-win):

            lphist = []
            for j in range(len(t)):
                for hw in range(-win, win+1):
                    for hh in range(-win, win+1):

                        if I[x+hw, y+hh] > (I[x, y]+t[j]):
                            sp[hw+win][hh+win] = 1
                        elif (I[x, y]-t[j]) <= I[x+hw, y+hh] <= (I[x, y]+t[j]):
                            se[hw+win][hh+win] = 1
                        elif I[x+hw, y+hh] < (I[x, y]-t[j]):
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
        print('running...', x)
    featuredata = np.vstack(lpdata)
    return featuredata


LPH = lph()    # lph should be calculated for 100 coordinates

# print(LPH)
print(LPH.shape)


'''
# saving training features in pickle
with open('test_data.dat', "wb") as f:
    pickle.dump(LPH, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('test_data.dat', 'rb') as f:
    print(pickle.load(f))
'''
np.save("test_data_by_numpy", LPH)
# loaded_by_numpy = np.load("test_data_by_numpy.npy")
# print(loaded_by_numpy.shape)


