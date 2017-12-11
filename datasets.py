import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import skimage.io as io


#   #########################################################################
common_path = '/run/media/manuel/Data/Manuel/OCT/TiSapphire/aidata/'


dict_glass = {'TiSaManu/170525_134751_manu_forearm1_01/': [None, None],
              'TiSaManu/170525_135539_manu_forearm1_02/': [None, None],
              'TiSaManu/170525_141326_manu_forearm2_01/': [1, 20],
              'TiSaManu/170525_141326_manu_forearm2_01b/': [2, 34],#[2+1444, 34+1444],
              'TiSaManu/170525_143005_manu_forearm2_02/': [1, 50],
              'TiSaManu/170525_144242_manu_forearm3_01/': [1, 72],
              'TiSaManu/170525_145009_manu_forearm3_02/': [1, 91],
              'TiSaManu/170525_150448_manu_indexF1_02/': [1, 70],
              'TiSaManu/170525_151532_manu_indexF2_01/': [1, 71],
              'TiSaManu/170525_152513_manu_RindexStrange_01/': [None, None],
              'TiSaManu/170525_153203_manu_RindexStrange_02/': [None, None],
              'TiSaManu/170525_153727_manu_ForearmL1_01/': [1, 37],
              'TiSaManu/170525_154517_manu_ForearmL1_02/': [None, None],
              'TiSaManu/170525_171603_manu_ForearmL1_03/': [None, None],
              'TiSaManu/170525_172725_manu_ForearmL1_04/': [1, 21],
              'TiSaManu/170525_173330_manu_ForearmL1_05/': [None, None],
              'TiSaManu/170525_174324_manu_ForearmL1_06/': [1, 42]
              # 'TiSaSou//170524_15_45_04_[40x_20cm_N32_fr180_W960_H960]_(sou_20x_0.8ms_01)/': [None, None],
              # 'TiSaSou//170524_15_52_25_[40x_20cm_N32_fr180_W960_H960]_(sou_20x_0.8ms_02)/': [None, None],
              # 'TiSaSou//170524_16_39_40_[40x_20cm_N32_fr179_W960_H960]_(sou_20x_1ms_01)/': [None, None],
              # 'TiSaSou//170524_17_22_48_[40x_20cm_N32_fr180_W960_H960]_(sou_20x_1ms_01)/': [None, None],
              # 'TiSaSou//170524_17_41_23_[40x_20cm_N32_fr180_W960_H960]_(sou_20x_1.1ms_01)/': [None, None],
              # 'TiSaSou//170524_18_22_15_[40x_20cm_N32_fr180_W960_H960]_(sou_20x_1.0ms_02)/': [None, None],
              # 'TiSaSou//170524_17_28_09_[40x_20cm_N32_fr180_W960_H960]_(Jheng-Yu_20x_1ms_01)/': [None, None],
              # 'TiSaSou//170524_17_33_18_[40x_20cm_N32_fr180_W960_H960]_(Jheng-Yu_20x_1ms_02)/': [None, None]
              }

dict_epidermis = {'TiSaManu/170525_134751_manu_forearm1_01/': [100, 195],
                  'TiSaManu/170525_135539_manu_forearm1_02/': [90, 230],
                  'TiSaManu/170525_141326_manu_forearm2_01/': [100, 330],
                  'TiSaManu/170525_141326_manu_forearm2_01b/': [120, 250],#[120+1444, 250+1444],
                  'TiSaManu/170525_143005_manu_forearm2_02/': [150, 300],
                  'TiSaManu/170525_144242_manu_forearm3_01/': [175, 270],
                  'TiSaManu/170525_145009_manu_forearm3_02/': [160, 300],
                  'TiSaManu/170525_150448_manu_indexF1_02/': [None, None],
                  'TiSaManu/170525_151532_manu_indexF2_01/': [None, None],
                  'TiSaManu/170525_152513_manu_RindexStrange_01/': [None, None],
                  'TiSaManu/170525_153203_manu_RindexStrange_02/': [None, None],
                  'TiSaManu/170525_153727_manu_ForearmL1_01/': [100, 240],
                  'TiSaManu/170525_154517_manu_ForearmL1_02/': [None, None],
                  'TiSaManu/170525_171603_manu_ForearmL1_03/': [80, 180],
                  'TiSaManu/170525_172725_manu_ForearmL1_04/': [75, 150],
                  'TiSaManu/170525_173330_manu_ForearmL1_05/': [None, None],
                  'TiSaManu/170525_174324_manu_ForearmL1_06/': [110, 230]
                  # 'TiSaSou//170524_15_45_04_[40x_20cm_N32_fr180_W960_H960]_(sou_20x_0.8ms_01)/': [1, 70],
                  # 'TiSaSou//170524_15_52_25_[40x_20cm_N32_fr180_W960_H960]_(sou_20x_0.8ms_02)/': [None, None],
                  # 'TiSaSou//170524_16_39_40_[40x_20cm_N32_fr179_W960_H960]_(sou_20x_1ms_01)/': [None, None],
                  # 'TiSaSou//170524_17_22_48_[40x_20cm_N32_fr180_W960_H960]_(sou_20x_1ms_01)/': [None, 37],
                  # 'TiSaSou//170524_17_41_23_[40x_20cm_N32_fr180_W960_H960]_(sou_20x_1.1ms_01)/': [None, None],
                  # 'TiSaSou//170524_18_22_15_[40x_20cm_N32_fr180_W960_H960]_(sou_20x_1.0ms_02)/': [None, None],
                  # 'TiSaSou//170524_17_28_09_[40x_20cm_N32_fr180_W960_H960]_(Jheng-Yu_20x_1ms_01)/': [None, None],
                  # 'TiSaSou//170524_17_33_18_[40x_20cm_N32_fr180_W960_H960]_(Jheng-Yu_20x_1ms_02)/': [None, 90]
                  }

dict_papillae = {'TiSaManu/170525_134751_manu_forearm1_01/': [230, 370],
                 'TiSaManu/170525_135539_manu_forearm1_02/': [250, 320],
                 'TiSaManu/170525_141326_manu_forearm2_01/': [260, 300],
                 'TiSaManu/170525_141326_manu_forearm2_01b/': [290, 340],#[290+1444, 340+1444],
                 'TiSaManu/170525_143005_manu_forearm2_02/': [325, 400],
                 'TiSaManu/170525_144242_manu_forearm3_01/': [310, 390],
                 'TiSaManu/170525_145009_manu_forearm3_02/': [340, 400],
                 'TiSaManu/170525_150448_manu_indexF1_02/': [None, None],
                 'TiSaManu/170525_151532_manu_indexF2_01/': [None, None],
                 'TiSaManu/170525_152513_manu_RindexStrange_01/': [None, None],
                 'TiSaManu/170525_153203_manu_RindexStrange_02/': [None, None],
                 'TiSaManu/170525_153727_manu_ForearmL1_01/': [250, 350],
                 'TiSaManu/170525_154517_manu_ForearmL1_02/': [None, None],
                 'TiSaManu/170525_171603_manu_ForearmL1_03/': [240, 320],
                 'TiSaManu/170525_172725_manu_ForearmL1_04/': [None, None],
                 'TiSaManu/170525_173330_manu_ForearmL1_05/': [None, None],
                 'TiSaManu/170525_174324_manu_ForearmL1_06/': [None, None]
                 # 'TiSaSou//170524_15_45_04_[40x_20cm_N32_fr180_W960_H960]_(sou_20x_0.8ms_01)/': [85, 310],
                 # 'TiSaSou//170524_15_52_25_[40x_20cm_N32_fr180_W960_H960]_(sou_20x_0.8ms_02)/': [None, None],
                 # 'TiSaSou//170524_16_39_40_[40x_20cm_N32_fr179_W960_H960]_(sou_20x_1ms_01)/': [126, None],
                 # 'TiSaSou//170524_17_22_48_[40x_20cm_N32_fr180_W960_H960]_(sou_20x_1ms_01)/':[70, 200],
                 # 'TiSaSou//170524_17_41_23_[40x_20cm_N32_fr180_W960_H960]_(sou_20x_1.1ms_01)/': [171, 320],
                 # 'TiSaSou//170524_18_22_15_[40x_20cm_N32_fr180_W960_H960]_(sou_20x_1.0ms_02)/': [None, None],
                 # 'TiSaSou//170524_17_28_09_[40x_20cm_N32_fr180_W960_H960]_(Jheng-Yu_20x_1ms_01)/': [None, None],
                 # 'TiSaSou//170524_17_33_18_[40x_20cm_N32_fr180_W960_H960]_(Jheng-Yu_20x_1ms_02)/': [150, 230]
                 }

dict_dermis = {'TiSaManu/170525_134751_manu_forearm1_01/': [450, 600],
               'TiSaManu/170525_135539_manu_forearm1_02/': [450, 600],
               'TiSaManu/170525_141326_manu_forearm2_01/': [360, 650],
               'TiSaManu/170525_141326_manu_forearm2_01b/': [420, 600],#[420+1444, 600+1444],
               'TiSaManu/170525_143005_manu_forearm2_02/': [410, 600],
               'TiSaManu/170525_144242_manu_forearm3_01/': [400, 700],
               'TiSaManu/170525_145009_manu_forearm3_02/': [500, 800],
               'TiSaManu/170525_150448_manu_indexF1_02/': [None, None],
               'TiSaManu/170525_151532_manu_indexF2_01/': [None, None],
               'TiSaManu/170525_152513_manu_RindexStrange_01/': [None, None],
               'TiSaManu/170525_153203_manu_RindexStrange_02/': [None, None],
               'TiSaManu/170525_153727_manu_ForearmL1_01/': [400, 529],
               'TiSaManu/170525_154517_manu_ForearmL1_02/': [None, None],
               'TiSaManu/170525_171603_manu_ForearmL1_03/': [350, 700],
               'TiSaManu/170525_172725_manu_ForearmL1_04/': [None, None],
               'TiSaManu/170525_173330_manu_ForearmL1_05/': [None, None],
               'TiSaManu/170525_174324_manu_ForearmL1_06/': [500, 650]
               # 'TiSaSou//170524_15_45_04_[40x_20cm_N32_fr180_W960_H960]_(sou_20x_0.8ms_01)/': [340, 700],
               # 'TiSaSou//170524_15_52_25_[40x_20cm_N32_fr180_W960_H960]_(sou_20x_0.8ms_02)/': [None, None],
               # 'TiSaSou//170524_16_39_40_[40x_20cm_N32_fr179_W960_H960]_(sou_20x_1ms_01)/': [None, None],
               # 'TiSaSou//170524_17_22_48_[40x_20cm_N32_fr180_W960_H960]_(sou_20x_1ms_01)/': [400, 700],
               # 'TiSaSou//170524_17_41_23_[40x_20cm_N32_fr180_W960_H960]_(sou_20x_1.1ms_01)/': [420, 700],
               # 'TiSaSou//170524_18_22_15_[40x_20cm_N32_fr180_W960_H960]_(sou_20x_1.0ms_02)/': [450, 750],
               # 'TiSaSou//170524_17_28_09_[40x_20cm_N32_fr180_W960_H960]_(Jheng-Yu_20x_1ms_01)/': [None, None],
               # 'TiSaSou//170524_17_33_18_[40x_20cm_N32_fr180_W960_H960]_(Jheng-Yu_20x_1ms_02)/': [480, 800]
               }


dataset = {'glass': dict_glass,
           'epidermis': dict_epidermis,
           'papillae': dict_papillae,
           'dermis': dict_dermis}

labels = {'glass': np.array([[1, 0, 0, 0]]),
          'epidermis': np.array([[0, 1, 0, 0]]),
          'papillae': np.array([[0, 0, 1, 0]]),
          'dermis': np.array([[0, 0, 0, 1]])}


#   #########################################################################


# Load data
def clean_data(dict_data):
    new_dictionary = {}
    # nnn = 0

    for label in dict_data.keys():
        data = dict_data[label]
        new_dictionary[label] = {}

        for elem in data.keys():
            # print(elem)

            if None in data[elem]:
                continue
            else:
                # sub_dict = {elem: data[elem]}
                new_dictionary[label][elem] = data[elem]

    # for i in new_dictionary.keys():
    #     for j in new_dictionary[i].keys():
    #         nnn += 1
    # print('The loop went through... ', nnn)

    return new_dictionary


#   #########################################################################


def save_data(dict_data, path_to_data):

    np.save(path_to_data, dict_data)


#   #########################################################################


def load_data(data_set, labels):

    # data_set = np.load(data_set_file)#, mmap_mode='r')

    n_H, n_W, n_C = 240, 240, 1
    num_labels = len(labels)
    name_len = 5
    file_extension = '.tif'
    step = 5

    X_train = np.empty((0, n_H, n_W, n_C))
    Y_train = np.empty((0, num_labels))
    print(X_train.shape)
    print(Y_train.shape)
    nnn = 0

    for label in data_set.keys():
        print("Loading '%s' (%i folders)" % (label, len(data_set[label])))
        prev_ex = X_train.shape[0]

        for folder in data_set[label].keys():
            start, stop = data_set[label][folder]

            for elem in range(start, stop, step):
                path_to_img = common_path + folder + str(elem).zfill(name_len) + file_extension
                img = Image.open(path_to_img)
                img_np = np.array(img).reshape(1, 240, 240, 1)
                X_train = np.vstack((X_train, img_np))
                Y_train = np.vstack((Y_train, labels[label]))
                # img_array.append(np.array(img))
                # print(img_array)

        new_ex = X_train.shape[0] - prev_ex
        print("    --->  %i examples loaded" % new_ex)
        # X_train = np.vstack((X_train, np.array(img_array)))
        # Y_train = np.vstack((Y_train, np.repeat(labels[label], data_set[label].shape[0], axis=0)))

    return X_train, Y_train


#   #########################################################################


def evaluate_data_manually():
    show_random = np.random.randint(0, X_train.shape[0], 20)

    for i in show_random:
        plt.imshow(X_train[i, :][:, :, 0])
        plt.title(Y_train[i])
        print('Showing data %i' % i)
        plt.show()


#   #########################################################################


# path_to_data = './data/dict_data.npy'
#
#
# print("Cleaning data...")
# cleaned_data = clean_data(dataset)
# print("")
#
#
# print("Saving cleaned data...")
# save_data(cleaned_data, path_to_data)
# print("")
#
#
# print("Loading data...")
# X_train, Y_train = load_data(cleaned_data, labels)
# print("    X: ", X_train.shape)
# print("    Y: ", Y_train.shape)
# print("")
#
#
# print(" --- EVALUATE DATA --- ")
# evaluate_data_manually()
# print("")
#
#
# print("Saving data...")
# np.save('data/Xdata', X_train)
# np.save('data/Ydata', Y_train)
# print("")


#   #########################################################################


# print(" --- LOAD DATA --- ")
# Xdata = np.load('./data/Xdata.npy')
# Ydata = np.load('./data/Ydata.npy')
# print("")
#
#
# print("Randomize data...")
# pp = np.random.permutation(Xdata.shape[0])
# Xdata_random = Xdata[pp, :, :, :]
# Ydata_random = Ydata[pp, :]
# print("")
#
#
# print(" --- DEFINE SETS --- ")
# X_train = Xdata_random[0:600, :, :, :]
# Y_train = Ydata_random[0:600, :]
#
# X_dev = Xdata_random[600:800, :, :, :]
# Y_dev = Ydata_random[600:800, :]
#
# X_test = Xdata_random[800:, :, :, :]
# Y_test = Ydata_random[800:, :]
# print("")
#
#
# print(" --- EVALUATE DATA --- ")
# evaluate_data_manually()
# print("")
#
#
# print("Saving data...")
# np.save('data/X_train', X_train)
# np.save('data/Y_train', Y_train)
#
# np.save('data/X_dev', X_dev)
# np.save('data/Y_dev', Y_dev)
#
# np.save('data/X_test', X_test)
# np.save('data/Y_test', Y_test)
# print("")




