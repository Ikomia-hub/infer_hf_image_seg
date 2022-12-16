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
from ikomia.utils import pyqtutils, qtconversion
from infer_huggingface_image_segmentation.infer_huggingface_image_segmentation_process import InferHuggingfaceImageSegmentationParam
from torch.cuda import is_available
# PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferHuggingfaceImageSegmentationWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferHuggingfaceImageSegmentationParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.gridLayout = QGridLayout()

        # Cuda
        self.check_cuda = pyqtutils.append_check(
                        self.gridLayout, "Cuda",
                        self.parameters.cuda and is_available())

        # Model loading method
        self.combo_model = pyqtutils.append_combo(self.gridLayout, "Model (COCO Panoptic)")
        self.combo_model.addItem("From: Costum model name")
        self.combo_model.addItem("facebook/detr-resnet-50-panoptic")
        self.combo_model.addItem("facebook/detr-resnet-50-dc5-panoptic")
        self.combo_model.addItem("facebook/detr-resnet-101-panoptic")
  
        self.combo_model.setCurrentText(self.parameters.model_name)

        # Load manually selected model card
        self.load_model_card = pyqtutils.append_browse_file(
                                                self.gridLayout,
                                                "Model Name or Checkpoint path:",
                                                self.parameters.model_card
                                                )

        # Number of iteration
        self.conf_tresh = pyqtutils.append_double_spin(
                        self.gridLayout, "Confidence threshold:", self.parameters.tresh,
                        min = 0.01, max = 1, step = 0.01, decimals = 2)


        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.gridLayout)

        # Set widget layout
        self.setLayout(layout_ptr)

        self.combo_model.currentTextChanged.connect(self.on_combo_task_changed)
        self.load_model_card.setVisible(self.combo_model.currentText() == "From: Costum model name")

    def on_combo_task_changed(self):
        self.load_model_card.setVisible(self.combo_model.currentText() == "From: Costum model name")

    def onApply(self):
        # Apply button clicked slot
        self.parameters.update = True
        self.parameters.model_name = self.combo_model.currentText()
        self.parameters.model_card = self.load_model_card.path
        self.parameters.cuda = self.check_cuda.isChecked()
        self.parameters.tresh = self.conf_tresh.value()

        # Send signal to launch the process
        self.emitApply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferHuggingfaceImageSegmentationWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "infer_huggingface_image_segmentation"

    def create(self, param):
        # Create widget object
        return InferHuggingfaceImageSegmentationWidget(param, None)
