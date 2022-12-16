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
import transformers
from transformers import AutoFeatureExtractor, AutoModelForImageSegmentation
from transformers.models.detr.feature_extraction_detr import rgb_to_id
from detectron2.data import MetadataCatalog
import numpy as np
import torch
import random
import io
from PIL import Image
import cv2


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
        self.model_card = "facebook/detr-resnet-50-panoptic"
        self.update = False
        self.tresh = 0.5

    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.cuda = strtobool(param_map["cuda"])
        self.model_name = str(param_map["model_name"])
        self.model_card = str(param_map["model_card"])
        self.update = strtobool(param_map["update"])
        self.tresh = param_map["conf_tresh"]

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = core.ParamMap()
        param_map["cuda"] = str(self.cuda)
        param_map["model_name"]= str(self.model_name)
        param_map["model_card"] = str(self.model_card)
        param_map["update"] = str(self.update)
        param_map["conf_tresh"] = int(self.tresh)
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferHuggingfaceImageSegmentation(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        # Add input/output of the process here
        self.setOutputDataType(core.IODataType.IMAGE_LABEL, 0)
        self.addOutput(dataprocess.CImageIO())
        self.addOutput(dataprocess.CNumericIO())
        self.addOutput(dataprocess.CGraphicsOutput())

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

    def infer(self, img, graphics_output):

        param = self.getParam()

        # Image pre-pocessing (image transformation and conversion to PyTorch tensor)
        encoding  = self.feature_extractor(img, return_tensors="pt")
        if param.cuda is True:
            #encoding = encoding.cuda()
            encoding = encoding.to(self.device)
        # Prediction
        outputs = self.model(**encoding)

        processed_sizes = torch.as_tensor(encoding['pixel_values'].shape[-2:]).unsqueeze(0)

        result = self.feature_extractor.post_process_panoptic(
                                                            outputs,
                                                            processed_sizes,
                                                            threshold=param.tresh
                                                            )[0]


        # Segments info and the panoptic result from DETR's prediction
        segments_info = deepcopy(result["segments_info"])

        # Panoptic predictions are stored in a special format png
        masks = Image.open(io.BytesIO(result['png_string']))
  
        # We convert the png into an segment id map
        h, w, c = np.shape(img)
        masks = np.array(masks, dtype=np.uint8)
        masks = cv2.resize(masks, (w,h), interpolation = cv2.INTER_NEAREST)
        masks = torch.from_numpy(rgb_to_id(masks))

        panoptic_seg = torch.full((h, w), fill_value=0)

        # Convertion of the class id
        for i in range(len(segments_info)):
            c = segments_info[i]["category_id"]
            segments_info[i]["category_id"] = self.meta.thing_dataset_id_to_contiguous_id[c]\
            if segments_info[i]["isthing"] else self.meta.stuff_dataset_id_to_contiguous_id[c]

        for info in segments_info:
            offset = len(self.thing_classes) if not info["isthing"] else 0
            px_value = info["id"]
            cat_value = info["category_id"]
            bool_mask = masks == px_value
            y, x = np.median(bool_mask.cpu().numpy().nonzero(), axis=1)
            properties_text = core.GraphicsTextProperty()
            properties_text.color = self.colors[cat_value+offset]
            properties_text.font_size = 7
            graphics_output.addText(self.classes[offset + cat_value], x, y, properties_text)
            panoptic_seg[bool_mask] = cat_value+1+offset
        return panoptic_seg.cpu().numpy()

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()

        self.forwardInputImage(0, 1)

        param = self.getParam()

        if param.update or self.model is None:
            # Feature extractor selection
            if param.model_card == "":
                param.model_card = None
            if param.model_name == "From: Costum model name":
                self.model_id = param.model_card
            else:
                self.model_id = param.model_name
                param.model_card = None
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_id)

            # Loading model weight
            self.model = AutoModelForImageSegmentation.from_pretrained(self.model_id)
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
            self.colors = [[0,0,0]]+[[int(c[0]), int(c[1]), int(c[2])] for c in self.colors]
            self.setOutputColorMap(1, 0, self.colors)
     
            # Setting legend
            legend = self.getOutput(2)
            legend.addValueList(list(range(1,1+len(self.classes))), "Class value", self.classes)

            param.update = False

        #Get input
        input = self.getInput(0)

        # Get output
        graphics_output = self.getOutput(3)
        graphics_output.setImageIndex(1)
        graphics_output.setNewLayer("PanopticSegmentation")
        if input.isDataAvailable():
            img = input.getImage()
            output_mask = self.getOutput(0)
            out = self.infer(img, graphics_output)
            output_mask.setImage(out)

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()


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
