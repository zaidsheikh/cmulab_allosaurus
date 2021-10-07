#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

from allosaurus.app import read_recognizer
from allosaurus.bin.prep_feat import prepare_feature
from allosaurus.bin.prep_token import prepare_token
from allosaurus.bin.download_model import download_model
from allosaurus.model import copy_model, resolve_model_name
from allosaurus.am.factory import transfer_am
from allosaurus.am.trainer import Trainer
from allosaurus.am.loader import read_loader


recognizers = {}


def recognize(input_file, params={}):
   """
   params is a dictionary of key, value pairs
   """

   lang = params.get("lang", "ipa")
   model = params.get("model", "latest")

   if model not in recognizers:
      if resolve_model_name(model) == "none":
         download_model(model)
      recognizers[model] = read_recognizer(model)

   return recognizers[model].recognize(input_file, lang_id=lang)


def fine_tune(data_dir, pretrained_model, new_model_name, params={}):
   """
   params is a dictionary of key, value pairs
   data_dir should have the following structure
   data_dir/
   ├── train/
   │   ├── utt_1.wav
   │   ├── utt_1.wav
   │   └── ...
   └── validate/
       ├── utt_x.wav
       ├── utt_x.txt
       └── ...
   """

   default_params = {
      "pretrained_model": pretrained_model,
      "new_model": new_model_name,
      "path": data_dir,
      "lang": "ipa",
      "batch_frame_size": 6000,
      "criterion": "ctc",
      "optimizer": "sgd",
      "lr": 0.01,
      "grad_clip": 5.0,
      "epoch": 10,
      "log": "none",
      "verbose": True,
      "report_per_batch": 10,
      "device_id": -1
   }
   default_params.update(params)

   train_config = argparse.Namespace(**default_params)

   data_path = Path(train_config.path)
   for subdir in ["train", "validate"]:

      with open(data_path / subdir / 'wave', 'w') as fwave:
         for wavfile in sorted((data_path / subdir).glob("*.wav")):
            utt_id = wavfile.name[:-len(".wav")]
            fwave.write(utt_id + " " + str(wavfile.resolve()) + "\n")

      with open(data_path / subdir / 'text', 'w') as ftext:
         for txtfile in sorted((data_path / subdir).glob("*.txt")):
            utt_id = txtfile.name[:-len(".txt")]
            transcription = txtfile.read_text().replace('\n', ' ').strip()
            ftext.write(utt_id + " " + transcription + "\n")

      prepare_feature(data_path / subdir, train_config.pretrained_model)
      prepare_token(data_path / subdir, train_config.pretrained_model, train_config.lang)

   # prepare training and validating loaders
   train_loader = read_loader(data_path / 'train', train_config)
   validate_loader = read_loader(data_path / 'validate', train_config)

   # initialize the target model path with the old model
   copy_model(train_config.pretrained_model, train_config.new_model)
   model = transfer_am(train_config)

   # train
   trainer = Trainer(model, train_config)
   trainer.train(train_loader, validate_loader)

   train_loader.close()
   validate_loader.close()
