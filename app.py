import os
import random
import math

import pandas as pd
import numpy as np
import cv2 as cv
import torch
import segmentation_models_pytorch as smp
import albumentations as a
import matplotlib.pyplot as plt
import tifffile as tiff
import PySimpleGUI as sg


GPU = torch.cuda.is_available()

### Calculating the length per pixel
# Original size in pixels (1024, 1360, 3)
# Original size in micrometer (512, 680)
# 1 pixel = 0.5 micrometer
LENGTH_PER_PIXEL_ORIGINAL = (1024./512., 1360./680.)

# Cropped image has size of (600, 600, 3) 
# Resized image has size of (400, 400, 3)
# Scaled from 600 -> 400
LENGTH_PER_PIXEL_SCALED = tuple([x / 600. * 400. for x in LENGTH_PER_PIXEL_ORIGINAL])
SCALING_FACTOR = 400./600.

# Insert cell area threshold (in micrometer squared) here
# Cell groups having area greater than this value will not be identified
CELL_THRESHOLD = 2000

# Do not modify this line
# Convert mm to pixels
CELL_THRESHOLD *= 2**2

PARENT = os.path.dirname(os.getcwd())

CONSTANTS = [('Core Area', 'Area (μm²)', "core_areas"), ('Invasive Area', 'Area (μm²)', "invasive_areas"), ('Core Radius', 'Micrometer (μm)', "core_radii"), ('Invasive Radius', 'Micrometer (μm)', "invasive_radii"), ('Core Perimeter', 'Micrometer (μm)', "core_perimeters"), ('Invasive Perimeter', 'Micrometer (μm)', "invasive_perimeters"), ("Complexity", "Complexity", "complexities")]

DF_COLLECTION = []
COMBINED_DF_COLLECTION = []

def get_sharpness(path):
    img = cv.imread(path)
    laplacian = cv.Laplacian(img, cv.CV_64F)
    sharpness = laplacian.var()
    return sharpness
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # gray_f = float(gray)
    # diff = gray_f[:-1, :] - gray_f[1:, :]
    # return sum(np.abs(diff))


def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    if GPU: torch.cuda.manual_seed(seed)


# Locate timm-regnety_320-FPN_epoch_19_0.665670.pth
# Add as shortcut


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """

    _transform = [
        a.Lambda(image=preprocessing_fn),
        a.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return a.Compose(_transform)


def draw_on_spheroid(img, core_mask, invasive_mask, cell_mask):
    img = np.array(img, dtype=np.uint8)
    img = cv.resize(img, (1360, 1024))

    # img = cv.resize(img, (400, 400))
 

    core_mask = np.array(core_mask, dtype=np.uint8).squeeze()
    invasive_mask = np.array(invasive_mask, dtype=np.uint8).squeeze()
    cell_mask = np.array(cell_mask, dtype=np.uint8).squeeze()
    
    # Resize to original size
    core_mask = cv.resize(core_mask, (1360, 1024))
    invasive_mask = cv.resize(invasive_mask, (1360, 1024))
    cell_mask = cv.resize(cell_mask, (1360, 1024))


    core_mask = cv.bitwise_and(img, img, mask=core_mask)
    invasive_mask = cv.bitwise_and(img, img, mask=invasive_mask)
    cell_mask = cv.bitwise_and(img, img, mask=cell_mask)


    _, core_img_thresh = cv.threshold(core_mask, 1, 255, cv.THRESH_BINARY)
    _, invasive_img_thresh = cv.threshold(invasive_mask, 1, 255, cv.THRESH_BINARY)
    _, cell_img_thresh = cv.threshold(cell_mask, 1, 255, cv.THRESH_BINARY)

    # # Select only 1 channel
    core_img_thresh = core_img_thresh[:, :, 0]
    invasive_img_thresh = invasive_img_thresh[:, :, 0]
    cell_img_thresh = cell_img_thresh[:, :, 0]

    # Finding the contours
    core_cnts, _ = cv.findContours(core_img_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    core_cnts = sorted(core_cnts, key=cv.contourArea, reverse=True)
    invasive_cnts, _ = cv.findContours(invasive_img_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    invasive_cnts = sorted(invasive_cnts, key=cv.contourArea, reverse=True)
    cell_cnts, _ = cv.findContours(cell_img_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cell_cnts = [c for c in sorted(cell_cnts, key=cv.contourArea, reverse=True) if cv.contourArea(c) <= CELL_THRESHOLD]

    # Draw core_perimeter and invasive_perimeter
    cv.drawContours(img, core_cnts, 0, (0, 255, 0), 2)
    cv.drawContours(img, invasive_cnts, 0, (255, 255, 0), 2)
    cv.drawContours(img, cell_cnts, -1, (64, 224, 208), 2)

    
    core_area = cv.contourArea(core_cnts[0])
    invasive_area = cv.contourArea(invasive_cnts[0])
    cell_area = [cv.contourArea(c) for c in cell_cnts]

    core_perimeter = cv.arcLength(core_cnts[0], True)
    invasive_perimeter = cv.arcLength(invasive_cnts[0], True)
    cell_perimeter = [cv.arcLength(c, True) for c in cell_cnts]

    core_center, core_radius = cv.minEnclosingCircle(core_cnts[0])
    center, invasive_radius = cv.minEnclosingCircle(invasive_cnts[0])

    cv.circle(img, (int(center[0]), int(center[1])), int(invasive_radius), (255, 0, 0), 2)
    cv.circle(img, (int(core_center[0]), int(core_center[1])), int(core_radius), (0, 0, 255), 2)
    cell_center, cell_radius = [], []
    for c in cell_cnts:
        cc, cr = cv.minEnclosingCircle(c)
        cell_center.append(cc)
        cell_radius.append(cr)

    # Draw invasive radius and core radius
    for i in range(len(cell_center)):
        cv.circle(img, (int(cell_center[i][0]), int(cell_center[i][1])), int(cell_radius[i]), (255, 0, 255), 2)

    core_area_mm = core_area * scaling_factor ** 2
    invasive_area_mm = invasive_area * scaling_factor ** 2
    cell_area_mm = [c * scaling_factor ** 2 for c in cell_area]

    core_radius_mm = core_radius * scaling_factor
    invasive_radius_mm = invasive_radius * scaling_factor
    cell_radius_mm = [c * scaling_factor for c in cell_radius]

    core_perimeter_mm = core_perimeter * scaling_factor
    invasive_perimeter_mm = invasive_perimeter * scaling_factor
    cell_perimeter_mm = [c * scaling_factor for c in cell_perimeter]

    complexity = (invasive_perimeter_mm ** 2) / (4 * math.pi * invasive_area_mm)

    return img, core_area_mm, invasive_area_mm, cell_area_mm, core_radius_mm, invasive_radius_mm, cell_radius_mm, core_perimeter_mm, invasive_perimeter_mm, cell_perimeter_mm, complexity


def get_spheroid_calculations(core_model, invasive_model, cell_model, in_dir=None, out_dir=None,  img_type='tif', threshold=0.75, preprocessing_fn_b7=None, preprocessing_fn_fpn=None):
    assert img_type in ['tif', 'jpeg', 'jpg', 'png'], 'Not supported image type.'
    assert in_dir != out_dir, 'Same directory. Please specify another directory for out_dir.'
    assert out_dir is not None, 'Please specify output directory (out_dir).'

    core_model.eval()
    invasive_model.eval()
    cell_model.eval()

    ids = []
    core_areas = []
    invasive_areas = []
    cell_areas = []
    core_radii = []
    invasive_radii = []
    cell_radii = []
    core_perimeters = []
    invasive_perimeters = []
    cell_perimeters = []
    complexities = []

  # Directory of image
  # if in_dir is not None:
  
    files = os.listdir(in_dir) if os.path.isdir(in_dir) else [in_dir]

    for f in files:
        path = os.path.join(in_dir, f) if os.path.isdir(in_dir) else f
        id = f if os.path.isdir(in_dir) else path.split('/')[-1].split('.')[0]

        if id not in CHOSEN_IMAGES: continue

        try:
            if img_type == 'tif':
                img = tiff.imread(path)
            elif img_type in ['jpeg', 'jpg', 'png']:
                img = cv.imread(path)
                img = img[:, :, ::-1]
        except Exception as e:
            print(e)

        # ori_img is not resized
        img_b7 = img.copy()
        img_b7 = cv.resize(img_b7, (480, 480))
    

        img_b7 = (img_b7 - img_b7.mean()) / img_b7.std()
        img_b7 = preprocessing_fn_b7(img_b7)
        img_b7 = img_b7[:, :, 0][..., np.newaxis]
        img_b7 = torch.tensor(img_b7)
        img_b7 = img_b7.permute(2, 0, 1)
        img_b7 = img_b7.unsqueeze(0)
        img_b7 = (img_b7.to('cuda') if GPU else img_b7).float()

        # ori_img is not resized
        img_fpn = img.copy()
        img_fpn = cv.resize(img_fpn, (480, 480))
    

        img_fpn = (img_fpn - img_fpn.mean()) / img_fpn.std()
        img_fpn = preprocessing_fn_b7(img_fpn)
        img_fpn = img_fpn[:, :, 0][..., np.newaxis]
        img_fpn = torch.tensor(img_fpn)
        img_fpn = img_fpn.permute(2, 0, 1)
        img_fpn = img_fpn.unsqueeze(0)
        img_fpn = (img_fpn.to('cuda') if GPU else img_fpn).float()

        with torch.no_grad():
            core_mask = core_model(img_b7)
            invasive_mask = invasive_model(img_b7)
            cell_mask = cell_model(img_fpn)


        core_mask = core_mask.detach().cpu().squeeze().sigmoid()
        core_mask[core_mask >= threshold] = 1
        core_mask[core_mask < threshold] = 0

        invasive_mask = invasive_mask.detach().cpu().squeeze().sigmoid()
        invasive_mask[invasive_mask >= threshold] = 1
        invasive_mask[invasive_mask < threshold] = 0

        cell_mask = cell_mask.detach().cpu().squeeze().sigmoid()
        cell_mask[cell_mask >= threshold] = 1
        cell_mask[cell_mask < threshold] = 0

    
        os.makedirs(out_dir.replace(" ", "_"), exist_ok=True)
        out_path = f"{out_dir}/{f}".replace(" ", "_")

        img, core_area_mm, invasive_area_mm, cell_area_mm, core_radius_mm, invasive_radius_mm, cell_radius_mm, core_perimeter_mm, invasive_perimeter_mm, cell_perimeter_mm, complexity = draw_on_spheroid(img, core_mask, invasive_mask, cell_mask)

        if img_type == 'tif':
            tiff.imsave(out_path, img)
        else:
            plt.imsave(out_path, img)

        # plt.figure(figsize=(10, 7.5))
        # plt.title(id)
        # plt.imshow(img)
        # plt.axis("off")

        # plt.show()


        # Resize to original scale
        # img = cv.resize(img, (1360, 1024))
    
        ids.append(id)
        core_areas.append(core_area_mm)
        invasive_areas.append(invasive_area_mm)
        cell_areas.append(cell_area_mm)

        core_radii.append(core_radius_mm)
        invasive_radii.append(invasive_radius_mm)
        cell_radii.append(cell_radius_mm)

        core_perimeters.append(core_perimeter_mm)
        invasive_perimeters.append(invasive_perimeter_mm)
        cell_perimeters.append(cell_perimeter_mm)

        complexities.append(complexity)

    return pd.DataFrame({"ids":ids, "core_areas":core_areas, "invasive_areas":invasive_areas, "cell_areas":cell_areas, "core_radii":core_radii, "invasive_radii":invasive_radii, "cell_radii":cell_radii, "core_perimeters":core_perimeters, "invasive_perimeters":invasive_perimeters, "cell_perimeters":cell_perimeters, "complexities":complexities})
  

def load_model(model, model_file):
    state_dict = torch.load(model_file) if GPU else torch.load(model_file, map_location=torch.device('cpu'))
    if "model_state_dict" in state_dict.keys():
        state_dict = state_dict['model_state_dict']
        
    state_dict = {k[7:] if k.startswith('module.') else k : state_dict[k] for k in state_dict.keys()}
    
    model.load_state_dict(state_dict)
    model.eval()

    return model

seed_everything()

core_model = smp.UnetPlusPlus(
    encoder_name='timm-efficientnet-b7', 
    encoder_weights=None, 
    encoder_depth=5,
    decoder_channels=(256, 128, 64, 32, 16),
    classes=len(['cell']), 
    activation=torch.nn.PReLU,
    in_channels=1,
    decoder_attention_type='scse'
)

invasive_model = smp.UnetPlusPlus(
    encoder_name='timm-efficientnet-b7', 
    encoder_weights=None, 
    encoder_depth=5,
    decoder_channels=(256, 128, 64, 32, 16),
    classes=len(['cell']), 
    activation=torch.nn.PReLU,
    in_channels=1,
    decoder_attention_type='scse'
)

cell_model = smp.FPN(
    encoder_name='timm-regnety_320',
    encoder_weights=None,
    activation=torch.nn.PReLU,
    classes=len(['cell']),
    in_channels=1
)

# cell_model_weight = "/content/drive/MyDrive/timm-efficientnet-b7-unetplusplus_epoch_13_0.489685.pth"

MODELS = zip([core_model, invasive_model, cell_model], ["CORE", "INVASIVE", "SINGLE_CELL"])

for idx, (model, name) in enumerate(MODELS):
    sg.one_line_progress_meter('Initialization',
                            idx+1, 3,
                            orientation='h',
                            grab_anywhere=True,
                            bar_color=('teal', 'white'))
    model = load_model(model, f"./models/{name}.pth")

preprocessing_fn_b7 = smp.encoders.get_preprocessing_fn('timm-efficientnet-b7', 'noisy-student')
preprocessing_fn_fpn = smp.encoders.get_preprocessing_fn('timm-regnety_320', 'imagenet')

if GPU: core_model.to("cuda")
if GPU: invasive_model.to("cuda")
if GPU: cell_model.to("cuda")

# Resized_img_size / original_img_size
# scaling_factor = 400./600.
scaling_factor = 512./1024.


CHOSEN_SPHEROIDS = []

if __name__ == "__main__":

    sg.theme('DarkBlack1')     # Please always add color to your window
    # The tab 1, 2, 3 layouts - what goes inside the tab

    #menu_def = [["File", ["New", "Import", "Export", "Save", "Save As"]],
    #                ["Help", ["Settings", "About", "Exit"]]]

    c_layout = [[sg.Text('Select condition(s) to be included:')]] + [[sg.Check(c, key=idx, default=True) for idx, c in enumerate(os.listdir(f"{PARENT}/INPUT"))]]

    window = sg.Window('Condition Selection', [c_layout, [sg.Button('Submit', key='-SUBMIT-')]], no_titlebar=False)

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            quit()
        if event == '-SUBMIT-':
            c_values = values
            window.close()
            break

    counter = 0
    l = len([v for v in c_values if c_values[v]==True])

    sg.one_line_progress_meter("Individual Cell Analysis",
                                counter, l,
                                orientation='h',
                                grab_anywhere=True,
                                bar_color=('teal', 'white'))


    for index, condition in enumerate(os.listdir(f"{PARENT}/INPUT")):

        if index not in [v for v in c_values if c_values[v]==True]: continue
        CHOSEN_IMAGES = []
        INPUT_PATH = f"{PARENT}/INPUT/{condition}"
        OUTPUT_PATH = f"{PARENT}/OUTPUT/{condition.replace(' ', '_')}"

        s_layout = [[sg.Text('Select spheroid(s) to be included:')]] + [[sg.Check(s, key=idx, default=True) for idx, s in enumerate(os.listdir(f"{INPUT_PATH}"))]]

        window = sg.Window('Spheroid Selection', [s_layout, [sg.Button('Submit', key='-SUBMIT-')]], no_titlebar=False)

        while True:
            event, values = window.read()
            if event == sg.WIN_CLOSED:
                quit()
            if event == '-SUBMIT-':
                CHOSEN_SPHEROIDS.append(values)
                window.close()
                break


        for idx, file in enumerate(os.listdir(INPUT_PATH)):

            if idx not in [v for v in CHOSEN_SPHEROIDS[index] if CHOSEN_SPHEROIDS[index][v]==True]: continue

            dir = os.listdir(f"{INPUT_PATH}/{file}")
            for id in {d.split("Z")[0][:-1] for d in dir}:
                img = [img for img in dir if img.startswith(f"{id}_")]
                sharpness = [get_sharpness(f"{INPUT_PATH}/{file}/{img}") for img in img]
                CHOSEN_IMAGES.append(img[np.argmax(sharpness)])

        for idx, file in enumerate(os.listdir(INPUT_PATH)):
            #sg.one_line_progress_meter("Individual Cell Analysis",
            #                            idx+1, len(os.listdir(INPUT_PATH))+1,
            #                            orientation='h',
            #                            grab_anywhere=True,
            #                            bar_color=('teal', 'white'))

            

            # for id in {dir.split("Z")[0][:-1] for dir in os.listdir(f"{PARENT}/INPUT/{file}")}:
            #   img = [img for img in os.listdir(f"{PARENT}/INPUT/{file}") if img.startswith(id)]
            #   sharpness = [get_sharpness(cv.imread(img)) for img in img]
            #   CHOSEN_IMAGES.append(img[np.argmax(sharpness)])

            df = get_spheroid_calculations(core_model, invasive_model, cell_model, in_dir=f"{INPUT_PATH}/{file}", out_dir=f"{OUTPUT_PATH}/{file}", threshold=0.75, preprocessing_fn_b7=preprocessing_fn_b7, preprocessing_fn_fpn=preprocessing_fn_fpn, img_type='jpg')
            df['time'] = df['ids'].apply(lambda x : x.split('_')[1][1:]).astype(int)
            df = df.sort_values(by='time').reset_index(drop=True)

            df.to_csv(f"{OUTPUT_PATH}/{file.replace(' ', '_')}/{file}.csv", index=False)

            DF_COLLECTION.append(df)

            for title, label, header in CONSTANTS:
                #plt.locator_params(nbins=len(df))
                plt.plot(df.time, df[header])
                plt.title(title)
                plt.xlabel('Time ( x * 10 Minutes )')
                plt.ylabel(label)
                plt.savefig(f"{OUTPUT_PATH}/{file.replace(' ', '_')}/{header}.png")
                plt.clf()

        DF = pd.concat(DF_COLLECTION, axis=0).reset_index(drop=True).sort_values(['time', 'ids'])
        DF["ids"] = DF["ids"].apply(lambda x : x.split("_")[0])

        try:
            os.makedirs(f"{PARENT}/COMBINED_OUTPUT/{condition.replace(' ', '_')}", exist_ok = True)
        except OSError as e:
            print(e)
    
        DF.to_csv(f"{PARENT}/COMBINED_OUTPUT/{condition.replace(' ', '_')}/ALL_SPHEROID_CALCULATIONS.csv", index=False)
        COMBINED_DF_COLLECTION.append(DF)
        df = DF.groupby(["time"])

        for idx, (title, label, header) in enumerate(CONSTANTS):
            
            #ax = DF.pivot_table(index='time', values=header, aggfunc=np.mean).plot()
            #ax.locator_params(nbins=len(DF))
            #ax.set_title(title)
            #ax.set_xlabel('Time ( x * 10 Minutes )')
            #ax.set_ylabel(label)

            df = DF.groupby(["time"])

            val = df[header].std() / df[header].mean() < 1
            df = DF.loc[DF["time"].isin(list(val[val].index))].groupby(["time"])

            #df = df.filter(lambda x: (x[header].std() < 1.5 * IQR).values.tolist())

            plt.fill_between(list(val[val].index), df[header].mean()-df[header].std(), df[header].mean()+df[header].std(), alpha=0.2, label='standard deviation')
            plt.errorbar(list(val[val].index), df[header].mean(), df[header].std(), elinewidth=.1)
            plt.title(title)
            plt.xlabel('Time ( x * 10 Minutes )')
            plt.ylabel(label)
            plt.savefig(f"{PARENT}/COMBINED_OUTPUT/{condition.replace(' ', '_')}/{header}.png")
            plt.clf()

        counter += 1
        sg.one_line_progress_meter("Individual Cell Analysis",
                                    counter, l,
                                    orientation='h',
                                    grab_anywhere=True,
                                    keep_on_top=True,
                                    bar_color=('teal', 'white'))
    


    c_layout = [[sg.Text('Select condition(s) to be included:')]] + [[sg.Check(c, key=idx, default=True) for idx, c in [(idx, c) for (idx, c) in enumerate(os.listdir(f"{PARENT}/INPUT")) if idx in c_values]]]

    window = sg.Window('Condition Selection', [c_layout, [sg.Button('Submit', key='-SUBMIT-')]], no_titlebar=False)

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            quit()
        if event == '-SUBMIT-':
            c_values = values
            window.close()
            break

    #CHOSEN_SPHEROIDS = []

    for index, condition in enumerate(os.listdir(f"{PARENT}/INPUT")):
        if index not in [v for v in c_values if c_values[v]==True]: continue

        s_layout = [[sg.Text('Select spheroid(s) to be included:')]] + [[sg.Check(s, key=idx, default=True) for idx, s in [(idx, s) for (idx, s) in enumerate(os.listdir(f"{PARENT}/INPUT/{condition}")) if idx in CHOSEN_SPHEROIDS[index]]]]

        window = sg.Window('Spheroid Selection', [s_layout, [sg.Button('Submit', key='-SUBMIT-')]], no_titlebar=False)

        while True:
            event, values = window.read()
            if event == sg.WIN_CLOSED:
                quit()
            if event == '-SUBMIT-':
                CHOSEN_SPHEROIDS.append(values)
                window.close()
                break

    for idx, (title, label, header) in enumerate(CONSTANTS):

        #c_layout = [[sg.Text('Select condition(s) to be included:')]] + [[sg.Check(c, key=idx, default=True) for idx, c in [(idx, c) for (idx, c) in enumerate(os.listdir(f"{PARENT}/INPUT")) if idx in c_values]]]

        #window = sg.Window('Condition Selection', [c_layout, [sg.Button('Submit', key='-SUBMIT-')]], no_titlebar=False)

        #while True:
        #    event, values = window.read()
        #    if event == sg.WIN_CLOSED:
        #        quit()
        #    if event == '-SUBMIT-':
        #        c_values = values
        #        window.close()
        #        break

        counter = 0
        l = len([v for v in c_values if c_values[v]==True])

        sg.one_line_progress_meter("Combined Cell Analysis",
                                    counter, l,
                                    orientation='h',
                                    grab_anywhere=True,
                                    keep_on_top=True,
                                    bar_color=('teal', 'white'))

        for index, condition in enumerate(os.listdir(f"{PARENT}/INPUT")):
            if index not in [v for v in c_values if c_values[v]==True]: continue

        #    s_layout = [[sg.Text('Select spheroid(s) to be included:')]] + [[sg.Check(s, key=idx, default=True) for idx, s in [(idx, s) for (idx, s) in enumerate(os.listdir(f"{INPUT_PATH}")) if idx in s_values]]]

        #    window = sg.Window('Spheroid Selection', [s_layout, [sg.Button('Submit', key='-SUBMIT-')]], no_titlebar=False)

        #    while True:
        #        event, values = window.read()
        #        if event == sg.WIN_CLOSED:
        #            quit()
        #        if event == '-SUBMIT-':
        #            s_val
        #            
        #            ues = values
        #            window.close()
        #            break

            df = pd.read_csv(f"{PARENT}/COMBINED_OUTPUT/{condition.replace(' ', '_')}/ALL_SPHEROID_CALCULATIONS.csv").groupby(["time"])
            plt.fill_between(np.unique(COMBINED_DF_COLLECTION[index]["time"].values), df[header].mean()-df[header].std(), df[header].mean()+df[header].std(), alpha=0.2, label=f"{condition} Stdev")
            plt.errorbar(np.unique(COMBINED_DF_COLLECTION[index]["time"].values), df[header].mean(), df[header].std(), label=f"{condition} Mean", elinewidth=.1)
            plt.title(title)
            plt.xlabel('Time ( x * 10 Minutes )')
            plt.ylabel(label)
        plt.legend()
        plt.savefig(f"{PARENT}/COMBINED_OUTPUT/ALL_CONDITIONS/{header}.png")
        plt.clf()









# tmp = df.copy()
# fig, ax = plt.subplots(3, 3)


# ax[0][0].plot(tmp.time, tmp.core_areas)
# ax[0][1].plot(tmp.time, tmp.invasive_areas)
# ax[0][2].plot(tmp.time, tmp.core_radii)
# ax[1][0].plot(tmp.time, tmp.invasive_radii)
# ax[1][1].plot(tmp.time, tmp.core_perimeters)
# ax[1][2].plot(tmp.time, tmp.invasive_perimeters)
# ax[2][0].plot(tmp.time, tmp.complexities)

# ax[0][0].set_title('Core Area')
# ax[0][0].set_xlabel('Time ( x * 10 Minutes )')
# ax[0][0].set_ylabel('Area (μm²)')

# ax[0][1].set_title('Invasive Area')
# ax[0][1].set_xlabel('Time ( x * 10 Minutes )')
# ax[0][1].set_ylabel('Area (μm²)')

# ax[0][2].set_title('Core Radius')
# ax[0][2].set_xlabel('Time ( x * 10 Minutes )')
# ax[0][2].set_ylabel('Micrometer (μm)')

# ax[1][0].set_title('Invasive Radius')
# ax[1][0].set_xlabel('Time ( x * 10 Minutes )')
# ax[1][0].set_ylabel('Micrometer (μm)')

# ax[1][1].set_title('Core Perimeter')
# ax[1][1].set_xlabel('Time ( x * 10 Minutes )')
# ax[1][1].set_ylabel('Micrometer (μm)')

# ax[1][2].set_title('Invasive Perimeter')
# ax[1][2].set_xlabel('Time ( x * 10 Minutes )')
# ax[1][2].set_ylabel('Micrometer (μm)')

# ax[2][0].set_title('Complexity')
# ax[2][0].set_xlabel('Time ( x * 10 Minutes )')
# ax[2][0].set_ylabel('Complexity')

# plt.subplots_adjust(hspace=1.0, wspace=0.75)

# # plt.show()

# fig.savefig(fig_path)
