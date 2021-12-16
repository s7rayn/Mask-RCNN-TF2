import os, json, time
import xml
import numpy as np
from mrcnn import model as modellib, utils, visualize
from PIL import Image, ImageDraw
from samples.coco import coco
from mrcnn.config import Config
import skimage.draw

# Pocet tried v datasete
NUM_CLASSES = 1
#NUM_CLASSES = 13550

# Relativna cesta k .h5 suboru s vahami
WEIGHTS_FILE = 'mask_rcnn_coco.h5'

# Relativna cesta k JSON anotaciam (trenovacie data)
TRAIN_ANNOTATIONS_FILE = './train.json'

# Relativna cesta k adresaru so subormi v anotaciach
TRAIN_ANNOTATION_IMAGE_DIR = './forms'

# Relativna cesta k JSON anotaciam (validacne data)
VALIDATION_ANNOTATIONS_FILE = './val.json'

# Relativna cesta k adresaru so subormi v anotaciach
VALIDATION_ANNOTATION_IMAGE_DIR = './forms'

# Pocet trenovacich epoch
NUM_EPOCHS = 50
MODEL_NAME = 'model_mrcnn'

# Nastavenie ROOT_DIR premennej na rootovsky priecinok (mimo git repozitara MASK_RCNN)
ROOT_DIR = os.getcwd()

# Priecinok kde sa budu ukladat logy a natrenovane modely
MODEL_DIR = os.path.join(ROOT_DIR, 'logs')


class TrainConfig(coco.CocoConfig):
    # Rozpoznavacie meno
    NAME = MODEL_NAME

    # Trenovat sa bude jeden obrazok na jednej GPU (batch_size je 1 -> GPUs * images/GPU)
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Pocet tried (prida pozadie -> triedu 0)
    NUM_CLASSES = 1 + NUM_CLASSES

    # Minimalne a maximalne rozmery (cisla musia byt minimalne 6 krat delitelne cislom 2)
    IMAGE_MIN_DIM = 320 # 359 je velkost obrazka ale to nie je delitelne 6 krat dvojkou
    IMAGE_MAX_DIM = 512

    IMAGE_RESIZE_MODE = "square"
    #IMAGE_SHAPE = [512, 359, 3]

    # Pocet trenovacich epoch (pocet_vzoriek/batch_size; batch_size=pocet_tried)
    STEPS_PER_EPOCH = 1232
    VALIDATION_STEPS = 307

    # Matterport povodne pouzival resnet101
    BACKBONE = 'resnet101'

    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    TRAIN_ROIS_PER_IMAGE = 200
    MAX_GT_INSTANCES = 114
    POST_NMS_ROIS_INFERENCE = 1000
    POST_NMS_ROIS_TRAINING = 2000

    DETECTION_MAX_INSTANCES = 114
    DETECTION_MIN_CONFIDENCE = 0.1


#TrainConfig().display()


class CocoLikeDataset(utils.Dataset):
    def load_data(self, annotation_json, images_dir):
        # Nacitanie JSON-u zo suboru
        json_file = open(annotation_json)
        coco_json = json.load(json_file)
        json_file.close()

        # Pridanie nazvov tried
        source_name = 'coco_like'
        for category in coco_json['categories']:
            class_id = category['id']
            class_name = category['name']
            if class_id < 1:
                print('Chyba: Trieda s nazvom "{}" nemoze mat ID nizsie nez 1 (ID triedy 0 je rezervovane pre pozadie)'.format(class_name))
                return

            self.add_class(source_name, class_id, class_name)

        # Nacitanie vsetkych anotacii
        annotations = {}
        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)

        # Nacnie vsetkych obrazkov a ich pridanie do datasetu
        seen_images = {}
        for image in coco_json['images']:
            image_id = image['id']
            if image_id in seen_images:
                print('Varovanie: Preskakujem duplicitny obrazok s ID "{}"'.format(image))
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image['file_name']
                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print('Varovanie: Preskakujem obrazok s ID "{}" kvoli chybajucemu klucu "{}"'.format(image_id, key))

                image_path = os.path.abspath(os.path.join(images_dir, image_file_name))
                image_annotations = annotations[image_id]

                # Prida obrazok pomocou povodnej (base) metody
                self.add_image(
                    source=source_name,
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations
                )

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []

        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            for segmentation in annotation['segmentation']:
                mask_draw.polygon(segmentation, fill=1)
                bool_array = np.array(mask) > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)

        return mask, class_ids


dataset_train = CocoLikeDataset()
dataset_train.load_data(TRAIN_ANNOTATIONS_FILE, TRAIN_ANNOTATION_IMAGE_DIR)
dataset_train.prepare()

dataset_val = CocoLikeDataset()
dataset_val.load_data(VALIDATION_ANNOTATIONS_FILE, VALIDATION_ANNOTATION_IMAGE_DIR)
dataset_val.prepare()

# Vytvorenie trenovacieho modelu
model = modellib.MaskRCNN(mode='training', config=TrainConfig(), model_dir=MODEL_DIR)

if WEIGHTS_FILE is not None:
    #model.load_weights(WEIGHTS_FILE, by_name=True)
    model.load_weights(WEIGHTS_FILE, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"])
    print('Vahy z COCO datasetu uspesne nacitane')

start_train = time.time()
model.train(dataset_train, dataset_val, learning_rate=TrainConfig().LEARNING_RATE, epochs=NUM_EPOCHS, layers='heads')
end_train = time.time()
minutes = round((end_train - start_train) / 60, 2)
print(f'Trenovanie trvalo {minutes} minut')
