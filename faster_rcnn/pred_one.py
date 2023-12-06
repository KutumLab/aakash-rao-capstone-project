from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
import matplotlib.pyplot as plt
import cv2.cv2 as cv2

if __name__ == "__main__":
    test_data = [{'file_name': '.../image_1jpg',
                    'image_id': 10}]

    cfg = get_cfg()
    cfg.merge_from_file("model config")
    cfg.MODEL.WEIGHTS = cfg.OUTPUT_DIR + "/model_final.pth"
    predictor = DefaultPredictor(cfg)
    im = cv2.imread(test_data[0]["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                scale=0.5,
                instance_mode=ColorMode.IMAGE_BW)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    img = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_RGBA2RGB)
    plt.imsave('.../image_1.jpg', img) # path for saving image
    # plot original
    plt.imshow(img)