# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from ikomia import core, dataprocess
import copy
from copy import deepcopy
from ikomia.utils import strtobool
from transformers import AutoFeatureExtractor, AutoModelForImageSegmentation
from transformers.models.detr.feature_extraction_detr import rgb_to_id
from detectron2.data import MetadataCatalog
import numpy as np
import torch
import random
import io
from PIL import Image
import os


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferHuggingfaceImageSegmentationParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        #self.cuda = torch.cuda.is_available()
        self.cuda = True if torch.cuda.is_available() else False
        self.model_name = "facebook/detr-resnet-50-panoptic"
        self.checkpoint_path = ""
        self.checkpoint = False
        self.conf_thres = 0.5
        self.update = False

    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.cuda = strtobool(param_map["cuda"])
        self.model_name = str(param_map["model_name"])
        self.pretrained = strtobool(param_map["checkpoint"])
        self.checkpoint_path = param_map["checkpoint_path"]
        self.conf_thres = float(param_map["conf_thres"])
        self.update = strtobool(param_map["update"])

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = core.ParamMap()
        param_map["cuda"] = str(self.cuda)
        param_map["model_name"] = str(self.model_name)
        param_map["checkpoint"] = str(self.checkpoint)
        param_map["checkpoint_path"] = self.checkpoint_path
        param_map["conf_thres"] = str(self.conf_thres)
        param_map["update"] = str(self.update)
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferHuggingfaceImageSegmentation(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        # Add input/output of the process here
        self.addOutput(dataprocess.CInstanceSegIO())

        # Create parameters class
        if param is None:
            self.setParam(InferHuggingfaceImageSegmentationParam())
        else:
            self.setParam(copy.deepcopy(param))
        # Detect if we have a GPU available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_id = None
        self.feature_extractor = None
        self.colors = None
        self.meta = None
        self.stuff_classes = None
        self.thing_classes = None
        self.classes = None
        self.update = False

    def getProgressSteps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()

        self.forwardInputImage(0, 0)

        param = self.getParam()

        if param.update or self.model is None:
            model_id = None
            # Feature extractor selection
            if param.checkpoint is False:
                model_id = param.model_name
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
            else:
                feature_extractor_path = os.path.join(param.checkpoint_path, "preprocessor_config.json")
                model_id = param.checkpoint_path
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_path)

            # Loading model weight
            self.model = AutoModelForImageSegmentation.from_pretrained(model_id)
            self.device = torch.device("cuda") if param.cuda else torch.device("cpu")
            self.model.to(self.device)
            print("Will run on {}".format(self.device.type))

            # Getting classe name
            self.meta = MetadataCatalog.get("coco_2017_val_panoptic_separated")
            self.stuff_classes = self.meta.get("stuff_classes")
            self.thing_classes = self.meta.get("thing_classes")
            self.classes = self.thing_classes + self.stuff_classes

            # Setting color
            np.random.seed(10)
            self.colors = np.array(np.random.randint(0, 255, (len(self.classes), 3)))
            self.colors = [[int(c[0]), int(c[1]), int(c[2])] for c in self.colors]
            self.setOutputColorMap(0, 1, self.colors)

            param.update = False

        #Get input
        input = self.getInput(0)
        image = input.getImage()
        self.infer(image)

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()

    def infer(self, image):

        param = self.getParam()

        # Image pre-pocessing (image transformation and conversion to PyTorch tensor)
        encoding  = self.feature_extractor(image, return_tensors="pt")
        if param.cuda is True:
            encoding = encoding.to(self.device)
        # Prediction
        outputs = self.model(**encoding)
        processed_sizes = torch.as_tensor(encoding['pixel_values'].shape[-2:]).unsqueeze(0)

        h, w, c = np.shape(image)
        result = self.feature_extractor.post_process_panoptic(
                                                            outputs,
                                                            processed_sizes,
                                                            threshold=param.conf_thres,
                                                            target_sizes=[(h,w)]
                                                            )[0]

        result_with_score = self.feature_extractor.post_process_instance_segmentation(
                                                            outputs,
                                                            threshold=param.conf_thres,
                                                            target_sizes=[(h, w)],
                                                            return_coco_annotation=True,
                                                            )[0]["segments_info"]
        # Get score                                        
        scores = [r["score"] for r in result_with_score]

        # Segments info and the panoptic result from DETR's prediction
        segments_info = deepcopy(result["segments_info"])

        # Get output :
        h, w, _ = np.shape(image)
        instance_output = self.getOutput(1)
        instance_output.init("PanopticSegmentation", 0, w, h)

        # Panoptic predictions are stored in a special format png
        masks = Image.open(io.BytesIO(result['png_string']))

        # We convert the png into an segment id map
        masks = np.array(masks, dtype=np.uint8)
        mask_id = rgb_to_id(masks)
        unique_colors = np.unique(mask_id).tolist()

        masks_binary = np.zeros((h,w))
        mask_list = []
        for color in unique_colors:
            object_mask = np.where(mask_id == color, 1, 0)
            mask_list.append(object_mask)
            masks_binary = np.dstack([object_mask, masks_binary])

        # Get bounding boxes from masks
        boxes = []
        for i in range(masks_binary.shape[-1]):
            m = masks_binary[:, :, i]
            # Bounding box.
            horizontal_indicies = np.where(np.any(m, axis=0))[0]
            vertical_indicies = np.where(np.any(m, axis=1))[0]
            if horizontal_indicies.shape[0]:
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
            boxes.append([x1, y1, x2, y2])
        boxes = boxes[:-1]
        boxes.reverse()

        # Convertion of the class id
        for i in range(len(segments_info)):
            c = segments_info[i]["category_id"]
            segments_info[i]["category_id"] = self.meta.thing_dataset_id_to_contiguous_id[c]\
            if segments_info[i]["isthing"] else self.meta.stuff_dataset_id_to_contiguous_id[c]

        for info,b, s, ml in zip(segments_info, boxes, scores, mask_list):
            x_obj = float(b[0])
            y_obj = float(b[1])
            h_obj = (float(b[3]) - y_obj)
            w_obj = (float(b[2]) - x_obj)
            offset = len(self.thing_classes) if not info["isthing"] else 0
            thing_or_stuff = 0 if info["isthing"] else 1
            cat_value = info["category_id"]
            instance_output.addInstance(
                                    info["id"],
                                    thing_or_stuff,
                                    info["category_id"],
                                    self.classes[offset + cat_value],
                                    s,
                                    x_obj,
                                    y_obj,
                                    w_obj,
                                    h_obj,
                                    ml,
                                    self.colors[cat_value+offset])


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferHuggingfaceImageSegmentationFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_huggingface_image_segmentation"
        self.info.shortDescription = "Panoptic segmentation using models from Hugging Face."
        self.info.description = "This plugin proposes inference for panoptic segmentation"\
                                "using transformers models from Hugging Face. It regroups"\
                                "models covered by the Hugging Face class:"\
                                "<AutoModelForImageSegmentation>. Models can be loaded either"\
                                "from your fine-tuned model (local) or from the Hugging Face Hub."
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Segmentation"
        self.info.version = "1.0.0"
        self.info.iconPath = "icons/icon.png"
        self.info.authors = "Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond,"\
                            "Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault,"\
                            "Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer,"\
                            "Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu,"\
                            "Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame,"\
                            "Quentin Lhoest, Alexander M. Rush"
        self.info.article = "Huggingface's Transformers: State-of-the-art Natural Language Processing"
        self.info.journal = "EMNLP"
        self.info.license = "Apache License Version 2.0"
        # URL of documentation
        self.info.documentationLink = "https://www.aclweb.org/anthology/2020.emnlp-demos.6"
        # Code source repository
        self.info.repository = "https://github.com/huggingface/transformers"
        # Keywords used for search
        self.info.keywords = "instance, segmentation, inference, transformer,"\
                            "Hugging Face, Pytorch, Dert, resnet, Facebook"

    def create(self, param=None):
        # Create process object
        return InferHuggingfaceImageSegmentation(self.info.name, param)
