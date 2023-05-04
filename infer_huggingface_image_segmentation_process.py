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
        self.model_name_or_path = ""
        self.cuda = True if torch.cuda.is_available() else False
        self.model_name = "facebook/detr-resnet-50-panoptic"
        self.model_path = ""
        self.use_custom_model = False
        self.conf_thres = 0.5
        self.update = False

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.model_name_or_path = param_map["model_name_or_path"]
        self.cuda = strtobool(param_map["cuda"])
        self.model_name = str(param_map["model_name"])
        self.pretrained = strtobool(param_map["use_custom_model"])
        self.model_path = param_map["model_path"]
        self.conf_thres = float(param_map["conf_thres"])
        self.update = strtobool(param_map["update"])

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {}
        param_map["model_name_or_path"] = self.model_name_or_path
        param_map["cuda"] = str(self.cuda)
        param_map["model_name"] = str(self.model_name)
        param_map["use_custom_model"] = str(self.use_custom_model)
        param_map["model_path"] = self.model_path
        param_map["conf_thres"] = str(self.conf_thres)
        param_map["update"] = str(self.update)
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferHuggingfaceImageSegmentation(dataprocess.CInstanceSegmentationTask):

    def __init__(self, name, param):
        dataprocess.CInstanceSegmentationTask.__init__(self, name)
        # Create parameters class
        if param is None:
            self.set_param_object(InferHuggingfaceImageSegmentationParam())
        else:
            self.set_param_object(copy.deepcopy(param))
        # Detect if we have a GPU available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_id = None
        self.feature_extractor = None
        self.meta = None
        self.stuff_classes = None
        self.thing_classes = None
        self.classes = None
        self.update = False

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def rgb_to_id(self, color):
        if isinstance(color, np.ndarray) and len(color.shape) == 3:
            if color.dtype == np.uint8:
                color = color.astype(np.int32)
            return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
        return int(color[0] + 256 * color[1] + 256 * 256 * color[2])

    def run(self):
        # Core function of your process
        # Call begin_task_run for initialization
        self.begin_task_run()

        param = self.get_param_object()

        if param.update or self.model is None:
            model_id = None
            # Feature extractor selection
            if param.model_path != "":
                if os.path.isfile(param.model_path):
                    directory = os.path.dirname(param.model_path)
                    model_id = directory
                    param.use_custom_model = True
                else:
                    model_id = param.model_path
                    param.use_custom_model = True
            if param.model_name_or_path != "":
                if os.path.isfile(param.model_name_or_path):
                    directory = os.path.dirname(param.model_name_or_path)
                    model_id = directory
                    param.use_custom_model = True      
                if os.path.isdir(param.model_name_or_path):
                    model_id = param.model_name_or_path
                    param.use_custom_model = True

            if param.use_custom_model is False:
                model_id = param.model_name
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
            else:
                feature_extractor_path = os.path.join(param.model_path, 
                                                      "preprocessor_config.json")
                model_id = param.model_path
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
            self.set_names(self.classes)

            param.update = False

        #Get input
        input = self.get_input(0)
        image = input.get_image()
        self.infer(image)

        # Step progress bar:
        self.emit_step_progress()

        # Call end_task_run to finalize process
        self.end_task_run()

    def infer(self, image):

        param = self.get_param_object()

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

        outputs.logits = outputs.logits.cpu()
        outputs.pred_boxes = outputs.pred_boxes.cpu()
        outputs.pred_masks = outputs.pred_masks.cpu()
        outputs.last_hidden_state = outputs.last_hidden_state.cpu()
        outputs.encoder_last_hidden_state = outputs.encoder_last_hidden_state.cpu()

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

        # Panoptic predictions are stored in a special format png
        masks = Image.open(io.BytesIO(result['png_string']))

        # We convert the png into an segment id map
        masks = np.array(masks, dtype=np.uint8)
        mask_id = self.rgb_to_id(masks)
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
            boxes.insert(0, [x1, y1, x2, y2])
        boxes = boxes[1:]

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
            self.add_object(
                            info["id"],
                            thing_or_stuff,
                            offset + info["category_id"],
                            s,
                            x_obj,
                            y_obj,
                            w_obj if info["isthing"] else float(0),
                            h_obj if info["isthing"] else float(0),
                            ml,
                            )


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferHuggingfaceImageSegmentationFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_huggingface_image_segmentation"
        self.info.short_description = "Panoptic segmentation using models from Hugging Face. "
        self.info.description = "This plugin proposes inference for panoptic segmentation "\
                                "using transformers models from Hugging Face. It regroups "\
                                "models covered by the Hugging Face class: "\
                                "AutoModelForImageSegmentation. Models can be loaded either "\
                                "from your fine-tuned model (local) or from the Hugging Face Hub."
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Segmentation"
        self.info.version = "1.0.0"
        self.info.icon_path = "icons/icon.png"
        self.info.authors = "Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, "\
                            "Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, "\
                            "Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, "\
                            "Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, "\
                            "Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, "\
                            "Quentin Lhoest, Alexander M. Rush"
        self.info.article = "Huggingface's Transformers: State-of-the-art Natural Language Processing"
        self.info.journal = "EMNLP"
        self.info.license = "Apache License Version 2.0"
        # URL of documentation
        self.info.documentation_link = "https://www.aclweb.org/anthology/2020.emnlp-demos.6"
        # Code source repository
        self.info.repository = "https://github.com/huggingface/transformers"
        # Keywords used for search
        self.info.keywords = "instance, segmentation, inference, transformer,"\
                            "Hugging Face, Pytorch, Dert, resnet, Facebook"

    def create(self, param=None):
        # Create process object
        return InferHuggingfaceImageSegmentation(self.info.name, param)
